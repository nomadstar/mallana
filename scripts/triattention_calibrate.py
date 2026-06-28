#!/usr/bin/env python3
"""
TriAttention Calibration and Numerical Validation Script

Measures perplexity with and without KV-cache page eviction to validate H6.1:
  "TriAttention page eviction maintains 95% PPL quality at 50% physical page budget."

Usage example (RTX 2050, qwen2.5-coder-1.5b):
  python3 scripts/triattention_calibrate.py \
    --model qwen2.5-coder-1.5b-bf16.gguf \
    --corpus data/wikitext2-test.txt \
    --context-len 2048 \
    --page-budget 32 \
    --chunks 5 \
    --extra-args "-ngl 99"

Page budget math (pg_block_size = 32 tokens/page):
  ctx=2048 → 64 pages total;  50% budget = 32 pages = 1024 tokens
  ctx=4096 → 128 pages total; 50% budget = 64 pages = 2048 tokens

Download wikitext-2 corpus:
  python3 -c "
  import pandas as pd
  df = pd.read_parquet('https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet')
  text = chr(10).join(l for l in chr(10).join(df['text'].dropna()).splitlines() if l.strip())
  open('data/wikitext2-test.txt','w').write(text)"
"""

import argparse
import os
import sys
import subprocess
import re
import json
import tempfile
import shutil
import statistics


PPL_SANITY_MAX = 500  # anything above this indicates a broken run


def find_binary(custom_path=None):
    if custom_path:
        if os.path.exists(custom_path) and os.path.isfile(custom_path):
            return custom_path
        print(f"Error: Specified binary path not found: {custom_path}", file=sys.stderr)
        sys.exit(1)

    search_paths = [
        "./build/bin/llama-perplexity",
        "./build-tri/bin/llama-perplexity",
        "./build/bin/llama-cli",
        "./build-tri/bin/llama-cli",
    ]
    for path in search_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return path

    print("Error: Could not find llama-perplexity or llama-cli.", file=sys.stderr)
    print("Specify the path with --binary <path>.", file=sys.stderr)
    sys.exit(1)


def run_perplexity_run(cmd):
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        print("Warning: binary exited with non-zero code:", result.returncode, file=sys.stderr)

    _num = r"[-+]?(?:nan|inf|[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)"
    m = re.search(rf"Final estimate:\s*PPL\s*=\s*({_num})", combined, re.IGNORECASE)
    if m:
        return float(m.group(1)), combined

    m = re.search(rf"\bPPL\s*=\s*({_num})", combined, re.IGNORECASE)
    if m:
        return float(m.group(1)), combined

    return None, combined


def run_stage(binary, model, corpus_file, context_len, page_budget, runs, chunks, extra_args):
    """Run llama-perplexity `runs` times and return the mean PPL."""
    cmd = [binary]
    if "llama-cli" in os.path.basename(binary):
        cmd.append("-ppl")
    cmd.extend(["-m", model, "-f", corpus_file, "-c", str(context_len)])
    if chunks > 0:
        cmd.extend(["--chunks", str(chunks)])
    if extra_args:
        cmd.extend(extra_args.strip().split())
    if page_budget > 0:
        cmd.extend(["--triattention-page-budget", str(page_budget)])

    ppl_values = []
    for run_idx in range(1, runs + 1):
        label = f"page_budget={page_budget}" if page_budget > 0 else "baseline"
        print(f"  Run {run_idx}/{runs} [{label}]", flush=True)
        ppl, output = run_perplexity_run(cmd)
        if ppl is None:
            print("Error: Failed to parse perplexity from output.", file=sys.stderr)
            print("--- Last 20 lines ---", file=sys.stderr)
            print("\n".join(output.splitlines()[-20:]), file=sys.stderr)
            sys.exit(1)
        if ppl > PPL_SANITY_MAX:
            print(
                f"Error: PPL={ppl:.2f} exceeds sanity limit ({PPL_SANITY_MAX}). "
                "This usually means the corpus is wrong (e.g. README instead of wikitext-2). "
                "Use data/wikitext2-test.txt — see script header for download instructions.",
                file=sys.stderr,
            )
            print("--- Last 20 lines ---", file=sys.stderr)
            print("\n".join(output.splitlines()[-20:]), file=sys.stderr)
            sys.exit(1)
        print(f"    PPL = {ppl:.4f}", flush=True)
        ppl_values.append(ppl)

    return statistics.mean(ppl_values)


def main():
    parser = argparse.ArgumentParser(
        description="TriAttention Calibration — measure PPL with/without KV-cache page eviction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument(
        "--corpus", "--prompt-file",
        dest="corpus",
        required=True,
        help="Path to plain-text evaluation corpus (e.g. data/wikitext2-test.txt)",
    )
    parser.add_argument(
        "--page-budget",
        type=int,
        default=0,
        help="Physical page budget for eviction run (0 = skip eviction stage, just run baseline). "
             "For 50%% test at ctx=2048: --page-budget 32; at ctx=4096: --page-budget 64",
    )
    parser.add_argument(
        "--context-len", type=int, default=2048,
        help="Context length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--runs", type=int, default=2,
        help="Number of repetitions to average (default: 2)",
    )
    parser.add_argument(
        "--chunks", type=int, default=5,
        help="Max number of chunks to evaluate (default: 5; use -1 for full corpus)",
    )
    parser.add_argument("--binary", help="Explicit path to llama-perplexity binary")
    parser.add_argument(
        "--extra-args", default="-ngl 99 -fa off",
        help="Extra args forwarded to the binary (default: '-ngl 99 -fa off'). "
             "'-fa off' forces Phase-1 gather path which correctly propagates TriAttention "
             "eviction during perplexity evaluation. Phase-2 native paged FA has a known "
             "multi-sequence page-table bug that produces incorrect PPL with n_seq>1.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    # Warn if corpus looks wrong (very small or looks like a markdown file)
    corpus_size = os.path.getsize(args.corpus)
    if corpus_size < 100_000:
        print(
            f"Warning: corpus is only {corpus_size:,} bytes. "
            "PPL estimates from small corpora are unstable. "
            "Recommend wikitext-2 test (~1.2 MB).",
            file=sys.stderr,
        )
    with open(args.corpus, encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().strip()
    if first_line.startswith("#") or first_line.startswith("[!["):
        print(
            "Warning: corpus looks like a Markdown file (starts with '#' or '[!['). "
            "Use data/wikitext2-test.txt for valid PPL measurements.",
            file=sys.stderr,
        )

    binary_path = find_binary(args.binary)
    print(f"Binary : {binary_path}")
    print(f"Model  : {args.model}")
    print(f"Corpus : {args.corpus} ({corpus_size:,} bytes)")
    print(f"ctx={args.context_len}  page_budget={args.page_budget}  "
          f"runs={args.runs}  chunks={args.chunks if args.chunks > 0 else 'all'}")

    # Page math summary
    pg_block_size = 32
    total_pages = args.context_len // pg_block_size
    if args.page_budget > 0:
        budget_pct = args.page_budget / total_pages * 100
        print(f"Page math: {total_pages} pages total, budget={args.page_budget} ({budget_pct:.0f}%)")

    print("\n=== STAGE 1: Baseline (no eviction) ===", flush=True)
    baseline_ppl = run_stage(
        binary=binary_path,
        model=args.model,
        corpus_file=args.corpus,
        context_len=args.context_len,
        page_budget=0,
        runs=args.runs,
        chunks=args.chunks,
        extra_args=args.extra_args,
    )
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    if args.page_budget <= 0:
        print("\n--page-budget not set; skipping eviction stage. Pass --page-budget N to test H6.1.")
        results = {
            "baseline_ppl": baseline_ppl,
            "eviction_ppl": None,
            "page_budget": 0,
            "context_len": args.context_len,
            "total_pages": total_pages,
            "budget_pct": None,
            "delta_ppl": None,
            "quality_retention_pct": None,
            "h6_1_pass": None,
        }
    else:
        print(f"\n=== STAGE 2: Eviction (page_budget={args.page_budget}) ===", flush=True)
        eviction_ppl = run_stage(
            binary=binary_path,
            model=args.model,
            corpus_file=args.corpus,
            context_len=args.context_len,
            page_budget=args.page_budget,
            runs=args.runs,
            chunks=args.chunks,
            extra_args=args.extra_args,
        )
        print(f"Eviction PPL: {eviction_ppl:.4f}")

        # quality_retention: 100% = no degradation; <100% = degradation
        # Formula: baseline/eviction × 100 (lower eviction_ppl → closer to 100%)
        quality_retention_pct = (baseline_ppl / eviction_ppl) * 100.0 if eviction_ppl > 0 else 0.0
        delta_ppl = eviction_ppl - baseline_ppl
        budget_pct = args.page_budget / total_pages * 100
        h6_1_pass = quality_retention_pct >= 95.0

        results = {
            "baseline_ppl": baseline_ppl,
            "eviction_ppl": eviction_ppl,
            "page_budget": args.page_budget,
            "context_len": args.context_len,
            "total_pages": total_pages,
            "budget_pct": budget_pct,
            "delta_ppl": delta_ppl,
            "quality_retention_pct": quality_retention_pct,
            "h6_1_pass": h6_1_pass,
        }

    output_dir = "research/milestone-007"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "calibration_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults written to {json_path}")

    # ASCII summary
    print("\n" + "=" * 72)
    print(f" {'TRIATTENTION CALIBRATION — H6.1 VALIDATION':^70}")
    print("=" * 72)
    print(f" Model  : {os.path.basename(args.model)}")
    print(f" Corpus : {os.path.basename(args.corpus)}")
    print(f" ctx={args.context_len}  total_pages={total_pages}  page_budget={args.page_budget}  runs={args.runs}")
    print("-" * 72)
    print(f" Baseline PPL  : {baseline_ppl:.4f}")
    if results["eviction_ppl"] is not None:
        print(f" Eviction PPL  : {results['eviction_ppl']:.4f}  (Δ = {results['delta_ppl']:+.4f})")
        print(f" Budget        : {args.page_budget}/{total_pages} pages = {results['budget_pct']:.0f}% of context")
        print(f" Retention     : {results['quality_retention_pct']:.2f}%  (target ≥ 95%)")
        h = "PASS ✓" if results["h6_1_pass"] else "FAIL ✗"
        print(f" H6.1          : {h}")
    print("=" * 72)

    if results.get("h6_1_pass") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
