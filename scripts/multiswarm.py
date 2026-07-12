#!/usr/bin/env python3
"""
Multi-Agent Orchestrator (Multiswarm) for llama-cpp-turboquant.
Orchestrates agy (Gemini/Antigravity), a configurable implementer (copilot by
default; claude or agy also supported), and opencode to iteratively plan,
implement, validate, and critique codebase improvements.

Roles:
- Architect (agy): High context capacity, designs the plan.
- Implementer (copilot by default; --implementer claude|agy|copilot): writes the code.
- Verifier/Critic (opencode): Reviews the diff and test logs, provides critique.
"""

import argparse
import json
import os
import re
import sys
import subprocess
import shutil
import threading
import time
from datetime import datetime

# ANSI Color codes
BOLD = "\033[1m"
GREEN = "\033[38;5;82m"
BLUE = "\033[38;5;39m"
YELLOW = "\033[38;5;214m"
RED = "\033[38;5;196m"
CYAN = "\033[38;5;51m"
RESET = "\033[0m"

# Temp files (ignored by .gitignore)
PLAN_FILE = ".multiswarm_plan.md"
SUMMARY_FILE = ".multiswarm_summary.md"
CRITIQUE_FILE = ".multiswarm_critique.md"
AUDIT_FILE = ".multiswarm_audit.md"
RECON_FILE = ".multiswarm_recon.md"
VALIDATION_LOG = ".multiswarm_validation.log"
HISTORY_LOG = ".multiswarm_history.log"

HEARTBEAT_INTERVAL = 30  # seconds of silence before printing a status line
BUILD_POLL_INTERVAL = 3  # seconds between /proc scans for live build/compile jobs

# This repo is a large llama.cpp fork (tens of thousands of functions). It has a
# pre-built, indexed code knowledge graph available via the `codebase-memory-mcp`
# MCP server, which is far cheaper in tokens than raw grep/find for locating code,
# tracing callers/callees, or understanding structure. Every delegated agent gets
# reminded of this so none of them burns its budget re-discovering the codebase
# with brute-force search.
CODEBASE_MEMORY_HINT = (
    "MANDATORY — CODE NAVIGATION FOR THIS REPO: this is a large llama.cpp fork already indexed in "
    "the `codebase-memory-mcp` knowledge graph as project 'home-ignatus-GitHub-mallana' (37k+ "
    "nodes, 155k+ edges). You MUST use its MCP tools as your PRIMARY means of navigating code: "
    "search_graph (find functions/classes/routes), trace_path (callers/callees & data flow — e.g. "
    "trace the llama_decode / first-ubatch path), get_code_snippet (read a symbol by "
    "qualified_name), search_code (graph-augmented grep), get_architecture, query_graph (Cypher, "
    "incl. complexity/hot-path metrics). Do NOT brute-force grep/find across the tree — it is slow "
    "and token-expensive here, and it is the wrong tool. Fall back to grep/Read ONLY for plain "
    "text, configs, or non-code files. Always pass project='home-ignatus-GitHub-mallana'."
)

# agy's own `--print` mode has a hardcoded 5-minute default wait (`--print-timeout`,
# see `agy --help`), independent of anything happening in this script. Any task
# involving a multi-minute CUDA compile or deep reasoning routinely exceeds that,
# so agy kills its own turn (nonzero exit) even though real work is still in
# flight underneath. Give it a much longer budget by default.
AGY_PRINT_TIMEOUT_DEFAULT = "25m"

# Process names that anchor a build tree (used to detect "the agent kicked off a
# build" even after the agent's own CLI process has exited/crashed/timed out).
BUILD_ROOT_NAMES = {"cmake", "make", "ninja"}
# Actual compiler/linker workers whose cmdline names the source file being built.
COMPILE_NAMES = {"cc1plus", "cc1", "nvcc", "ccache", "cicc", "ptxas",
                  "ld", "ld.bfd", "ld.gold", "ld.lld"}
_SRC_FILE_RE = re.compile(r'([\w./+-]+\.(?:cu|cpp|cc|cxx|c))(?:\s|$)')


def _read_ppid(pid):
    with open(f"/proc/{pid}/stat") as f:
        stat = f.read()
    # format: pid (comm) state ppid ...  -- comm may contain spaces/parens,
    # so split from the last ')' rather than by naive whitespace splitting.
    rest = stat[stat.rfind(')') + 2:].split()
    return int(rest[1])


def _read_comm(pid):
    with open(f"/proc/{pid}/comm") as f:
        return f.read().strip()


def _read_cmdline(pid):
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        raw = f.read()
    return raw.replace(b"\x00", b" ").decode(errors="replace").strip()


def _children_map():
    m = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            ppid = _read_ppid(pid)
        except (IOError, OSError, IndexError, ValueError):
            continue
        m.setdefault(ppid, []).append(pid)
    return m


def _descendants(root_pid, cmap=None):
    cmap = cmap if cmap is not None else _children_map()
    seen = set()
    frontier = [root_pid]
    while frontier:
        p = frontier.pop()
        for c in cmap.get(p, []):
            if c not in seen:
                seen.add(c)
                frontier.append(c)
    return seen


def _pid_alive(pid):
    return os.path.exists(f"/proc/{pid}")


def _scan_build_state(root_pid):
    """Inspect the live process tree rooted at root_pid for build activity.

    Reads directly from /proc, so it reflects reality even if the agent whose
    subprocess this is has already exited, crashed, or silently stopped
    narrating (e.g. it timed out waiting on its own `cmake --build` call).
    Returns (build_root_pids, active_source_files).
    """
    descendants = _descendants(root_pid)
    build_roots = set()
    active_files = set()
    for pid in descendants:
        try:
            comm = _read_comm(pid)
        except (IOError, OSError):
            continue
        if comm in BUILD_ROOT_NAMES:
            build_roots.add(pid)
        if comm in COMPILE_NAMES:
            try:
                cmdline = _read_cmdline(pid)
            except (IOError, OSError):
                continue
            matches = _SRC_FILE_RE.findall(cmdline)
            if matches:
                active_files.add(os.path.basename(matches[-1]))
    return build_roots, active_files


def check_cli_tool(name):
    """Check if a CLI tool is available in the system PATH."""
    return shutil.which(name) is not None

def format_cmd_display(cmd):
    """Return a human-readable command string, replacing long --prompt/--print values."""
    result = []
    skip_next = False
    for part in cmd:
        if skip_next:
            result.append("<prompt>")
            skip_next = False
        elif part in ("--prompt", "--print"):
            result.append(part)
            skip_next = True
        else:
            result.append(part)
    return " ".join(result)

def log_session(message):
    """Write log messages to history file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_LOG, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_with_output(cmd, prefix="", print_func=print, log_file=None, timeout=None):
    """Run a command streaming stdout/stderr in real time with a heartbeat.

    Prints a status line every HEARTBEAT_INTERVAL seconds when the subprocess
    produces no output, so the operator can tell it is still alive.
    Optionally tee's output to log_file (an open file object).
    If timeout (seconds) is set, the process is terminated once wall-clock
    elapsed exceeds it — used to bound a flaky, non-blocking helper (e.g. agy
    recon) so a hang cannot stall the pipeline.
    Returns the Popen object (with .returncode set).
    """
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        text=True,
    )
    # stdout is guaranteed non-None because we passed stdout=subprocess.PIPE above;
    # assert it so static type-checkers (ty) don't flag the .readline() access below.
    assert process.stdout is not None
    prefix_str = f"{BOLD}[{prefix}]{RESET} " if prefix else ""
    start = time.monotonic()
    last_output = [start]
    stop_event = threading.Event()
    detected_build_roots = set()
    last_active_files = set()

    def heartbeat():
        while not stop_event.is_set():
            if stop_event.wait(timeout=BUILD_POLL_INTERVAL):
                break
            # Read the real process tree from /proc — this is independent of
            # whatever the agent itself chooses to narrate on stdout, so it
            # still shows real compiler activity even if the agent goes quiet
            # (or crashes) while a build it launched keeps running.
            try:
                roots, files = _scan_build_state(process.pid)
            except Exception:
                roots, files = set(), set()
            detected_build_roots.update(roots)

            if files != last_active_files:
                last_active_files.clear()
                last_active_files.update(files)
                if files:
                    shown = ", ".join(sorted(files)[:4])
                    more = "" if len(files) <= 4 else f" (+{len(files) - 4} more)"
                    print_func(
                        f"{prefix_str}{CYAN}[build] {len(files)} job(s) compiling: "
                        f"{shown}{more}{RESET}",
                        flush=True,
                    )

            silence = time.monotonic() - last_output[0]
            if silence >= HEARTBEAT_INTERVAL:
                elapsed = time.monotonic() - start
                print_func(
                    f"{prefix_str}{YELLOW}⟳ still running… "
                    f"{elapsed:.0f}s elapsed, last output {silence:.0f}s ago{RESET}",
                    flush=True,
                )

            if timeout is not None and (time.monotonic() - start) > timeout:
                print_func(
                    f"{prefix_str}{RED}⏱ timeout ({timeout:.0f}s) exceeded — terminating "
                    f"(non-blocking helper; pipeline continues).{RESET}",
                    flush=True,
                )
                try:
                    process.terminate()
                except Exception:
                    pass
                break

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()

    for line in iter(process.stdout.readline, ""):
        if line:
            last_output[0] = time.monotonic()
            print_func(f"{prefix_str}{line}", end="", flush=True)
            if log_file:
                log_file.write(line)
                log_file.flush()

    # Final synchronous scan right as the process exits, closing the race window
    # where a build gets spawned in the last <BUILD_POLL_INTERVAL seconds before
    # the agent's own process exits: once the parent is gone, any orphaned child
    # is immediately reparented (to init/subreaper) by the kernel, so walking the
    # process tree from `process.pid` afterwards would never find it again. The
    # periodic heartbeat thread alone can miss that window; this catches it.
    try:
        roots, _files = _scan_build_state(process.pid)
        detected_build_roots.update(roots)
    except Exception:
        pass

    stop_event.set()
    hb.join()
    process.wait()

    # The agent process may exit (finish, crash, or time out internally) while
    # a build it launched is still compiling in the background — this happened
    # with agy timing out on its own `cmake --build` call. Don't let the swarm
    # move on to the next phase (or misreport the outcome) while that build is
    # still running: wait here until every detected build root actually exits.
    for root_pid in sorted(detected_build_roots):
        if not _pid_alive(root_pid):
            continue
        print_func(
            f"{prefix_str}{YELLOW}⏸ agent process exited but build (pid {root_pid}) "
            f"is still running — pausing before continuing...{RESET}",
            flush=True,
        )
        waited = 0
        while _pid_alive(root_pid):
            time.sleep(BUILD_POLL_INTERVAL)
            waited += BUILD_POLL_INTERVAL
            if waited % HEARTBEAT_INTERVAL == 0:
                _, files = _scan_build_state(root_pid)
                if files:
                    shown = ", ".join(sorted(files)[:4])
                    print_func(
                        f"{prefix_str}{CYAN}[build] still running ({waited}s): "
                        f"{shown}{RESET}",
                        flush=True,
                    )
        print_func(
            f"{prefix_str}{GREEN}[build] background build (pid {root_pid}) finished.{RESET}",
            flush=True,
        )

    elapsed = time.monotonic() - start
    rc_color = GREEN if process.returncode == 0 else RED
    rc_sym = "✔" if process.returncode == 0 else "✘"
    print_func(
        f"{prefix_str}{rc_color}{rc_sym} finished in {elapsed:.1f}s "
        f"(exit {process.returncode}){RESET}",
        flush=True,
    )
    return process

def run_planning(task, iteration, critique=None, model=None, skip_permissions=False, resume=False,
                  agy_print_timeout=AGY_PRINT_TIMEOUT_DEFAULT):
    """Phase 1: Architect (agy) designs or refines the plan."""
    if iteration == 1:
        prompt = (
            f"You are the ARCHITECT (Gemini / Antigravity CLI). The task is to: '{task}'.\n"
            f"The codebase is located in the current working directory. You can find key Flash Attention files "
            f"such as 'fattn-vec.cuh' and 'fattn-tile.cuh' in 'ggml/src/ggml-cuda/'. Do NOT run slow global searches "
            f"starting from '/' or '$HOME'. Analyze the codebase and write a precise, step-by-step implementation plan.\n"
            f"{CODEBASE_MEMORY_HINT}\n"
            f"Your plan MUST begin with YAML frontmatter (delimited by ---) containing:\n"
            f"- task: {task}\n"
            f"- created_at: <current datetime>\n"
            f"- scope: <e.g. feature, bugfix, tests-only, refactor>\n"
            f"- forbidden_paths: <list of file/directory paths the plan must not modify>\n"
            f"\n"
            f"Your output MUST be written in markdown and saved to the file '{PLAN_FILE}'.\n"
            f"CRITICAL: Do NOT run background/asynchronous commands or end your turn to wait for tasks. "
            f"Run all commands synchronously or read files directly. You MUST write the '{PLAN_FILE}' file in the same turn "
            f"before completing your execution. Do not write the implementation code itself, only the files to modify, "
            f"changes required, and tests needed. Avoid conversational filler."
        )
    else:
        prompt = (
            f"You are the ARCHITECT (Gemini / Antigravity CLI). The implementation failed validation.\n"
            f"Here is the critique/error log:\n"
            f"```\n{critique}\n```\n"
            f"Please read the current plan in '{PLAN_FILE}', refine the plan to resolve these errors, "
            f"and update the file '{PLAN_FILE}' with the refined plan. Avoid conversational filler.\n"
            f"IMPORTANT: Preserve the YAML frontmatter at the top of the plan. Update 'created_at' "
            f"to the current datetime if desired.\n"
            f"CRITICAL: Do NOT run background/asynchronous commands or end your turn to wait for tasks. "
            f"Run all commands synchronously or read files directly. You MUST write the updated '{PLAN_FILE}' file in the same turn "
            f"before completing your execution."
        )

    cmd = ["agy"]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    if resume and iteration > 1:
        cmd.append("--continue")
    cmd.extend(["--print-timeout", agy_print_timeout])

    cmd.extend(["--prompt", prompt])

    print(f"\n{BOLD}{CYAN}=== Phase 1: Planning with Architect (agy) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting planning phase, iteration {iteration}")

    # Run with real-time streaming output
    res = run_with_output(cmd, prefix="AGY")

    if res.returncode == 0:
        if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
            print(f"{YELLOW}Warning: '{PLAN_FILE}' was not created or is empty. Please verify the agent wrote it.{RESET}")
            return False
        return True
    else:
        print(f"{RED}Architect planning failed with return code {res.returncode}.{RESET}")
        return False

def run_recon(task, model=None, skip_permissions=False, timeout=300):
    """Phase 0 (optional): agy does a READ-ONLY reconnaissance pass.

    agy has high context capacity but is unreliable at synchronous execution (it tends to launch
    builds and hang). So it is used here purely as an advisory analyst: read the code, rank the
    likely root-cause sites, and hand the implementer a head start via RECON_FILE. This is
    strictly NON-BLOCKING — it is bounded by `timeout` and, if it produces nothing, the pipeline
    proceeds without it. It must never gate the run.
    """
    prompt = (
        f"You are the RECON ANALYST (Gemini/Antigravity — high context capacity). The team will "
        f"implement this task: '{task}'.\n"
        f"{CODEBASE_MEMORY_HINT}\n"
        f"Do a strictly READ-ONLY reconnaissance pass to give the implementer a head start. "
        f"CRITICAL: you MUST NOT run builds, tests, or ANY shell command that compiles or executes "
        f"code — only read files and reason. Do NOT background commands or end your turn to wait. "
        f"In a SINGLE turn, write '{RECON_FILE}' (markdown) with:\n"
        f"1. Ranked most-likely root-cause locations (file:line) with a one-line rationale each.\n"
        f"2. The key code excerpts the implementer needs (so it need not re-search).\n"
        f"3. How the relevant code likely differs from upstream llama.cpp.\n"
        f"4. Concrete first steps for the fix.\n"
        f"Be concise and specific. You MUST write '{RECON_FILE}' before ending your turn."
    )
    cmd = ["agy"]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    # Short print-timeout AND a hard wall-clock timeout below — agy must not stall the run.
    cmd.extend(["--print-timeout", "5m", "--prompt", prompt])

    print(f"\n{BOLD}{CYAN}=== Phase 0: Reconnaissance with agy (read-only, non-blocking) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting recon pass (timeout={timeout}s)")

    try:
        res = run_with_output(cmd, prefix="RECON", timeout=timeout)
    except Exception as e:
        print(f"{YELLOW}Recon pass errored ({e}); proceeding without it.{RESET}")
        return False

    if res.returncode == 0 and os.path.exists(RECON_FILE) and os.path.getsize(RECON_FILE) > 0:
        print(f"{GREEN}✔ Recon notes written to '{RECON_FILE}'.{RESET}")
        return True
    print(f"{YELLOW}Recon produced no usable notes (timeout/hang/empty); "
          f"proceeding without it — this never blocks the pipeline.{RESET}")
    return False


def run_implementation(task, iteration, model=None, skip_permissions=False, resume=False, implementer="claude",
                        agy_print_timeout=AGY_PRINT_TIMEOUT_DEFAULT, impl_timeout=None):
    """Phase 2: Implementer writes code changes based on the plan."""
    if not os.path.exists(PLAN_FILE):
        print(f"{RED}Error: Plan file '{PLAN_FILE}' not found!{RESET}")
        return False

    with open(PLAN_FILE, "r") as f:
        plan_content = f.read()

    critique_content = ""
    if os.path.exists(CRITIQUE_FILE):
        with open(CRITIQUE_FILE, "r") as f:
            critique_content = f.read()

    recon_content = ""
    if os.path.exists(RECON_FILE):
        with open(RECON_FILE, "r") as f:
            recon_content = f.read()

    prompt = (
        f"You are the IMPLEMENTER ({implementer}). Your task is to execute the implementation plan.\n"
        f"{CODEBASE_MEMORY_HINT}\n"
        f"Here is the plan:\n"
        f"```markdown\n{plan_content}\n```\n"
    )
    if recon_content:
        prompt += (
            f"A read-only RECON ANALYST produced these advisory notes (suspected root-cause sites, "
            f"code excerpts, upstream-diff hints). Use them as a head start to avoid re-searching, "
            f"but VERIFY before trusting them — they are hints, not ground truth:\n"
            f"```markdown\n{recon_content}\n```\n"
        )
    if critique_content:
        prompt += (
            f"IMPORTANT: The previous attempt failed validation. Here is the feedback/errors:\n"
            f"```\n{critique_content}\n```\n"
            f"Please address these errors and modify the implementation accordingly.\n"
        )
    prompt += (
        f"Please perform the required file edits/creations to implement the plan. "
        f"When you are finished, write a short summary of the files modified and changes made "
        f"to '{SUMMARY_FILE}' (e.g. 'Modified src/main.cpp to support X').\n"
        f"NEVER run interactive or stdin-blocking commands from your shell — they hang your turn "
        f"forever with no way to recover, and nothing can unblock you. In particular: do NOT run "
        f"`llama-cli` in conversation/chat mode (the `-cnv` flag), and do NOT start any REPL, "
        f"pager, editor, or a foreground server you then wait on. To exercise TEXT GENERATION, run "
        f"`MODEL=<model.gguf> bash scripts/gen-smoke.sh` — it starts a server, sends NON-interactive "
        f"requests, checks the output is coherent, and cleans up for you; use it instead of driving "
        f"`llama-cli`/`llama-server` by hand. If you must invoke a model binary directly, you MUST "
        f"pass non-interactive flags (`-no-cnv`, `--simple-io`) AND redirect stdin from /dev/null "
        f"(append `< /dev/null`), and never leave a server running.\n"
        f"CRITICAL: This is a single non-interactive turn (headless/print mode) — there is no "
        f"follow-up turn, scheduled wakeup, or notification that will ever resume you after this "
        f"response ends. Do NOT background a build/test/long-running command and then end your "
        f"turn 'waiting' for it to finish — that command will be orphaned and nothing will ever "
        f"read its result. Run builds, compiles, and validation commands SYNCHRONOUSLY in the "
        f"foreground (even if a CUDA build takes 5-10 minutes): if your shell/Bash tool accepts an "
        f"explicit timeout parameter, set it to the maximum allowed (e.g. 600000ms / 10 minutes) "
        f"for that one call instead of relying on its default (often ~2 minutes, which is shorter "
        f"than a CUDA rebuild) — do NOT pass any 'run in background' option for it. The call should "
        f"simply block until the build finishes; that blocking IS the pause, and your turn resumes "
        f"the instant the command returns, in the same response. Only after that command has "
        f"actually returned should you write '{SUMMARY_FILE}' and end your turn."
    )

    # Per-tool CLI conventions:
    #   claude  : --print <prompt>  , auto-approve via --dangerously-skip-permissions
    #   agy     : --prompt <prompt> , auto-approve via --dangerously-skip-permissions, --print-timeout
    #   copilot : --prompt <prompt> , auto-approve via --allow-all (tools+paths+urls; required
    #             for non-interactive mode so it can edit files and run commands unattended)
    if implementer == "claude":
        prompt_flag = "--print"
    else:
        prompt_flag = "--prompt"

    cmd = [implementer]
    if skip_permissions:
        if implementer == "copilot":
            cmd.append("--allow-all")
        else:
            cmd.append("--dangerously-skip-permissions")
    if model:
        cmd.extend(["--model", model])
    if resume and iteration > 1:
        cmd.append("--continue")
    if implementer == "agy":
        cmd.extend(["--print-timeout", agy_print_timeout])
    if implementer == "copilot":
        cmd.append("--no-color")

    cmd.extend([prompt_flag, prompt])

    print(f"\n{BOLD}{CYAN}=== Phase 2: Implementation with Implementer ({implementer}) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting implementation phase, iteration {iteration}")

    res = run_with_output(cmd, prefix=implementer.upper(), timeout=impl_timeout)
    if res.returncode != 0:
        return False
    if not os.path.exists(SUMMARY_FILE) or os.path.getsize(SUMMARY_FILE) == 0:
        print(f"{YELLOW}Warning: '{SUMMARY_FILE}' was not created or is empty.{RESET}")
        return False
    return True

def run_validation():
    """System runs local compilation & unit tests validation."""
    print(f"\n{BOLD}{CYAN}=== Phase 3: Verification (Local Validation) ==={RESET}")
    log_session("Running validation suite")

    validate_script = "scripts/validate.sh"
    if not os.path.exists(validate_script):
        print(f"{YELLOW}Warning: '{validate_script}' not found. Falling back to build tools...{RESET}")
        if os.path.exists("CMakeLists.txt") or os.path.exists("build"):
            cmd = ["cmake", "--build", "build"]
        else:
            return False, "No validation script or build system found."
    else:
        cmd = ["bash", validate_script]

    print(f"Executing validation: {' '.join(cmd)}")

    with open(VALIDATION_LOG, "w") as log_file:
        res = run_with_output(cmd, prefix="BUILD", log_file=log_file)

    log_content = ""
    if os.path.exists(VALIDATION_LOG):
        with open(VALIDATION_LOG, "r") as f:
            log_content = f.read()

    passed = (res.returncode == 0)
    if passed:
        print(f"{GREEN}✔ Local validation tests passed.{RESET}")
    else:
        print(f"{RED}✘ Local validation tests failed.{RESET}")

    return passed, log_content

def run_critique(task, iteration, validation_passed, validation_log, model=None, agent=None, resume=False):
    """Phase 4: Reviewer/Critic (opencode) reviews implementation and log."""
    diff_output = ""
    try:
        diff_res = subprocess.run(["git", "diff"], capture_output=True, text=True)
        diff_output = diff_res.stdout
    except Exception as e:
        diff_output = f"Could not run git diff: {str(e)}"

    prompt = (
        f"You are the VERIFIER & CRITIC (OpenCode). A set of changes has been implemented for task: '{task}'.\n"
        f"{CODEBASE_MEMORY_HINT}\n"
        f"Here is the local validation pass status: {validation_passed}\n"
        f"Here is the local validation log:\n"
        f"```\n{validation_log[:4000]}\n```\n"
    )
    if diff_output:
        prompt += (
            f"Here is the git diff of the current implementation:\n"
            f"```diff\n{diff_output[:8000]}\n```\n"
        )
    prompt += (
        f"Analyze the implementation, diff, and validation log against requirements.\n"
        f"If the compilation succeeds, unit tests pass, and the changes are completely correct, "
        f"write EXACTLY the word 'SUCCESS' in the file '{CRITIQUE_FILE}' and nothing else.\n"
        f"If there are compiler errors, unit test regressions, or code quality issues, "
        f"write a detailed markdown critique list in '{CRITIQUE_FILE}' listing what needs to be fixed. "
        f"Do NOT write 'SUCCESS' if there is any issue."
    )

    cmd = ["opencode", "run", prompt]
    if model:
        cmd.extend(["--model", model])
    if agent:
        cmd.extend(["--agent", agent])
    if resume and iteration > 1:
        cmd.append("--continue")

    print(f"\n{BOLD}{CYAN}=== Phase 4: Critique with Reviewer (opencode) ==={RESET}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting critique phase, iteration {iteration}")

    res = run_with_output(cmd, prefix="OPENCODE")
    if res.returncode != 0:
        return False
    if not os.path.exists(CRITIQUE_FILE) or os.path.getsize(CRITIQUE_FILE) == 0:
        print(f"{YELLOW}Warning: '{CRITIQUE_FILE}' was not created or is empty.{RESET}")
        return False
    return True

def run_audit(scope=None, model=None, agent=None, skip_permissions=False):
    """Audit-only mode: opencode reviews the codebase or recent changes.

    scope: optional string describing what to audit (e.g. "src/llama-kv-cache.cpp",
           "recent changes", "Phase 2 Flash Attention page table"). Defaults to git diff
           vs main branch.
    """
    diff_output = ""
    try:
        diff_res = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)
        staged_res = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
        diff_output = diff_res.stdout + staged_res.stdout
        if not diff_output.strip():
            log_res = subprocess.run(
                ["git", "diff", "origin/master...HEAD"],
                capture_output=True, text=True,
            )
            diff_output = log_res.stdout
    except Exception as e:
        diff_output = f"Could not run git diff: {e}"

    scope_desc = scope or "all recent changes (git diff vs HEAD and staged)"

    prompt = (
        f"You are the CODE AUDITOR for the llama-cpp-turboquant research project. "
        f"Your task is to audit: **{scope_desc}**.\n\n"
        f"{CODEBASE_MEMORY_HINT}\n\n"
        f"Focus areas:\n"
        f"1. **Correctness** — logic bugs, off-by-one errors, undefined behavior, incorrect math\n"
        f"2. **Security** — no hardcoded credentials, no command injection, no unvalidated inputs\n"
        f"3. **Performance** — unnecessary allocations, cache-unfriendly access patterns, GPU divergence\n"
        f"4. **Documentation** — are complex invariants explained? Are bugs/workarounds annotated?\n"
        f"5. **Test coverage** — are edge cases reachable by current tests?\n\n"
    )
    if diff_output.strip():
        prompt += (
            f"Here is the git diff to review:\n"
            f"```diff\n{diff_output[:12000]}\n```\n\n"
        )
    else:
        prompt += "No git diff available — review the overall codebase in the current directory.\n\n"
    prompt += (
        f"Write a structured markdown audit report to the file '{AUDIT_FILE}'. "
        f"Structure the report as:\n"
        f"# Code Audit Report\n"
        f"## Summary\n"
        f"## Findings (Severity: Critical / High / Medium / Low / Info)\n"
        f"## Recommendations\n\n"
        f"Be specific: include file names and line numbers where possible. "
        f"If the code is clean in a given area, say so explicitly."
    )

    cmd = ["opencode", "run", prompt]
    if model:
        cmd.extend(["--model", model])
    if agent:
        cmd.extend(["--agent", agent])

    print(f"\n{BOLD}{CYAN}=== AUDIT MODE: Code Review with opencode ==={RESET}")
    print(f"Scope: {scope_desc}")
    print(f"Executing: {format_cmd_display(cmd)}")
    log_session(f"Starting audit, scope={scope_desc!r}")

    res = run_with_output(cmd, prefix="OPENCODE")
    if res.returncode != 0:
        print(f"{RED}Audit failed (exit {res.returncode}).{RESET}")
        return False
    if not os.path.exists(AUDIT_FILE) or os.path.getsize(AUDIT_FILE) == 0:
        print(f"{YELLOW}Warning: '{AUDIT_FILE}' was not created or is empty.{RESET}")
        return False

    print(f"\n{BOLD}{GREEN}=== Audit Report ==={RESET}")
    with open(AUDIT_FILE) as f:
        print(f.read())
    return True


# Conclusions that count as a CI failure worth auto-fixing. "cancelled",
# "skipped" and "neutral" are not actionable; in-progress runs have no
# conclusion yet and are reported but not treated as failures.
CI_FAILURE_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "action_required"}
CI_LOG_TAIL_CHARS = 4000   # tail of --log-failed per run fed to the swarm
CI_MAX_FAILED_LOGS = 3     # cap on how many failed runs get their logs inlined


def git_current_branch():
    res = subprocess.run(["git", "branch", "--show-current"],
                         capture_output=True, text=True)
    return res.stdout.strip() or None


def get_ci_runs(branch=None, limit=20):
    """Fetch recent GitHub Actions runs for this repo via the gh CLI."""
    cmd = ["gh", "run", "list", "--limit", str(limit),
           "--json", "databaseId,workflowName,displayTitle,headBranch,event,"
                     "status,conclusion,url,createdAt"]
    if branch:
        cmd.extend(["--branch", branch])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"'gh run list' failed: {res.stderr.strip()}")
    return json.loads(res.stdout or "[]")


def get_failed_run_log(run_id, max_chars=CI_LOG_TAIL_CHARS):
    """Return the tail of the failed-step logs for a run (errors live at the end)."""
    res = subprocess.run(["gh", "run", "view", str(run_id), "--log-failed"],
                         capture_output=True, text=True)
    log = res.stdout if res.returncode == 0 else (res.stderr or "")
    log = log.strip()
    if len(log) > max_chars:
        log = "…(truncated)…\n" + log[-max_chars:]
    return log


def report_ci_status(branch=None, limit=20):
    """Print recent GitHub Actions runs and return the failed ones."""
    print(f"\n{BOLD}{CYAN}=== GitHub Actions Status ==={RESET}")
    scope = f"branch '{branch}'" if branch else "all branches"
    print(f"Scope: {scope} (last {limit} runs)")

    runs = get_ci_runs(branch, limit)
    if not runs:
        print(f"{YELLOW}No workflow runs found. Workflows may not be registered "
              f"yet (a push touching .github/workflows/ registers them).{RESET}")
        return [], []

    failed = []
    for run in runs:
        concl = run.get("conclusion") or run.get("status") or "?"
        if run.get("conclusion") in CI_FAILURE_CONCLUSIONS:
            color = RED
            failed.append(run)
        elif run.get("conclusion") == "success":
            color = GREEN
        else:
            color = YELLOW
        print(f"  {color}{concl:<16}{RESET} {run.get('workflowName', '?'):<28} "
              f"{run.get('headBranch', '?'):<24} {run.get('displayTitle', '')[:48]}")

    if failed:
        print(f"\n{RED}{len(failed)} failed run(s) detected.{RESET}")
    else:
        print(f"\n{GREEN}No failed runs.{RESET}")
    return failed, runs


def build_ci_fix_task(failed_runs):
    """Compose a swarm task from failed GitHub Actions runs, inlining failed logs."""
    lines = [
        "Fix the failing GitHub Actions workflows in this repository. "
        "Diagnose whether the root cause is in the workflow files "
        "(.github/workflows/) or in the code/tests they exercise, and apply a "
        "minimal fix. Failed runs:"
    ]
    for i, run in enumerate(failed_runs):
        lines.append(
            f"- {run.get('workflowName', '?')} — '{run.get('displayTitle', '')}' "
            f"on {run.get('headBranch', '?')} ({run.get('conclusion')}): {run.get('url', '')}"
        )
        if i < CI_MAX_FAILED_LOGS:
            log = get_failed_run_log(run.get("databaseId"))
            if log:
                lines.append(f"Failed-step log (tail):\n```\n{log}\n```")
    lines.append(
        "NOTE: GitHub Actions cannot be re-run locally; validate by making the "
        "local build and tests pass and, in the summary, explain why the CI "
        "failure is resolved. Do NOT push or trigger remote workflows — the "
        "repository owner pushes after review."
    )
    return "\n".join(lines)


def check_success():
    """Verify if the critique step declared success.

    Must be an exact match against the trimmed file content, per the critic
    prompt's own instruction ("write EXACTLY the word 'SUCCESS' ... and
    nothing else"). A substring check is unsafe: a critique that legitimately
    fails can still contain the word "SUCCESS" while explaining why it isn't
    one yet (e.g. "Do not write SUCCESS until X is fixed"), which would
    otherwise be misread as approval.
    """
    if not os.path.exists(CRITIQUE_FILE):
        return False
    with open(CRITIQUE_FILE, "r") as f:
        content = f.read().strip().upper()
    return content == "SUCCESS"

def parse_plan_task():
    """Parse the 'task' field from an existing plan file's YAML frontmatter."""
    if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
        return None
    with open(PLAN_FILE, "r") as f:
        content = f.read()
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            frontmatter = content[3:end].strip()
            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("task:"):
                    return line[len("task:"):].strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Swarm Orchestrator (Multiswarm)")
    parser.add_argument("--task", help="The development task or bug to fix (required unless --audit)")
    parser.add_argument("--audit", action="store_true",
                        help="Audit-only mode: skip planning/implementation, run opencode code review on "
                             "current diff and exit. Use --audit-scope to focus the review.")
    parser.add_argument("--audit-scope",
                        help="What to audit in --audit mode (e.g. 'src/llama-kv-cache.cpp Phase 2 FA fix'). "
                             "Defaults to all staged + unstaged changes.")
    parser.add_argument("--ci-status", action="store_true",
                        help="Print recent GitHub Actions run status (via gh) and exit. "
                             "Exit code 1 if there are failed runs.")
    parser.add_argument("--ci-fix", action="store_true",
                        help="Check GitHub Actions for failed runs; if any, auto-generate a fix task "
                             "from the failed logs and run the full swarm loop on it.")
    parser.add_argument("--ci-branch",
                        help="Branch filter for --ci-status/--ci-fix (default: all branches; "
                             "use 'current' for the checked-out branch)")
    parser.add_argument("--ci-limit", type=int, default=20,
                        help="How many recent runs to inspect in --ci-status/--ci-fix (default: 20)")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations/loops (default: 3)")
    parser.add_argument("--skip-permissions", action="store_true", help="Auto-approve tool permissions (skip prompts)")
    parser.add_argument("--no-interactive", action="store_true", help="Run without asking for confirmation between phases")
    parser.add_argument("--continue-session", action="store_true", help="Continue previous CLI sessions if possible")
    parser.add_argument("--use-plan", action="store_true", help="Use existing .multiswarm_plan.md without running Architect planning")
    parser.add_argument("--force-use-plan", action="store_true", help="Use existing plan even if its task field does not match --task")
    parser.add_argument("--implementer", default="copilot", choices=["claude", "agy", "copilot"],
                        help="Tool used as implementer in Phase 2 (default: copilot)")
    parser.add_argument("--recon", action="store_true",
                        help="Phase 0: run a read-only agy reconnaissance pass before implementation "
                             "(high context; non-blocking, bounded by --recon-timeout). Its notes seed "
                             "the implementer. A hang/timeout never stalls the run.")
    parser.add_argument("--recon-timeout", type=int, default=300,
                        help="Hard wall-clock timeout (seconds) for the --recon pass (default: 300)")
    parser.add_argument("--impl-timeout", type=int, default=1800,
                        help="Hard wall-clock timeout (seconds) for the implementer phase — a safety "
                             "net so a hang (e.g. the implementer launching an interactive command "
                             "that blocks on stdin) self-terminates instead of stalling forever. "
                             "Generous enough for a real edit+build. Set 0 to disable (default: 1800)")
    parser.add_argument("--agy-print-timeout", default=AGY_PRINT_TIMEOUT_DEFAULT,
                        help=f"agy --print-timeout value for planning/implementation phases "
                             f"(agy's own hardcoded default is 5m, too short for builds; default here: {AGY_PRINT_TIMEOUT_DEFAULT})")
    parser.add_argument("--model-agy", help="Model override for agy")
    parser.add_argument("--model-claude", help="Model override for claude (ignored when --implementer=agy)")
    parser.add_argument("--model-implementer", help="Model override for the implementer tool (takes precedence over --model-claude)")
    parser.add_argument("--model-opencode", help="Model override for opencode")
    parser.add_argument("--agent-opencode", help="Custom agent for opencode")
    parser.add_argument("--cleanup", action="store_true", help="Delete temporary multiswarm files on completion")
    args = parser.parse_args()

    # Audit-only mode: needs opencode only
    if args.audit:
        if not check_cli_tool("opencode"):
            print(f"{RED}Error: 'opencode' not found in PATH.{RESET}")
            sys.exit(1)
        print(f"{BOLD}{CYAN}================================================================{RESET}")
        print(f"{BOLD}{CYAN}                  MULTISWARM — AUDIT MODE                      {RESET}")
        print(f"{BOLD}{CYAN}================================================================{RESET}")
        ok = run_audit(
            scope=args.audit_scope,
            model=args.model_opencode,
            agent=args.agent_opencode,
            skip_permissions=args.skip_permissions,
        )
        sys.exit(0 if ok else 1)

    # CI modes: report GitHub Actions status; --ci-fix feeds failures into the swarm loop
    if args.ci_status or args.ci_fix:
        if not check_cli_tool("gh"):
            print(f"{RED}Error: 'gh' (GitHub CLI) not found in PATH.{RESET}")
            sys.exit(1)
        branch = args.ci_branch
        if branch == "current":
            branch = git_current_branch()
        try:
            failed_runs, _all_runs = report_ci_status(branch, args.ci_limit)
        except RuntimeError as e:
            print(f"{RED}Error: {e}{RESET}")
            sys.exit(1)
        if args.ci_status:
            sys.exit(1 if failed_runs else 0)
        # --ci-fix
        if not failed_runs:
            print(f"{GREEN}Nothing to fix — CI is green.{RESET}")
            sys.exit(0)
        if args.task:
            print(f"{YELLOW}Warning: --task is ignored with --ci-fix "
                  f"(task is generated from the CI failures).{RESET}")
        print("Collecting failed logs and generating fix task...")
        args.task = build_ci_fix_task(failed_runs)
        log_session(f"CI-fix mode: generated task from {len(failed_runs)} failed run(s)")

    if not args.task:
        parser.error("--task is required unless --audit is specified")

    # Validate that CLI tools are installed
    missing_tools = []
    for tool in ["agy", args.implementer, "opencode"]:
        if not check_cli_tool(tool):
            missing_tools.append(tool)

    if missing_tools:
        print(f"{RED}Error: The following required CLI tools are missing in the PATH: {', '.join(missing_tools)}{RESET}")
        print("Please ensure they are installed and in your environment PATH.")
        sys.exit(1)

    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f"{BOLD}{GREEN}                 MULTISWARM ORCHESTRATION ENGINE                {RESET}")
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print(f"Task: {args.task}")
    print(f"Max Iterations: {args.iterations}")
    print(f"Roles: Architect=agy | Implementer={args.implementer} | Critic=opencode")
    print(f"Skip Permissions: {args.skip_permissions}")
    print(f"Interactive Confirmation: {not args.no_interactive}")
    print(f"{BOLD}{GREEN}================================================================{RESET}\n")

    # Clear old temporary files from previous runs
    for f in [SUMMARY_FILE, CRITIQUE_FILE, VALIDATION_LOG, RECON_FILE]:
        if os.path.exists(f):
            os.remove(f)

    use_existing_plan = False
    if args.use_plan or args.force_use_plan:
        if not os.path.exists(PLAN_FILE) or os.path.getsize(PLAN_FILE) == 0:
            print(f"{RED}Error: --use-plan specified but '{PLAN_FILE}' does not exist or is empty!{RESET}")
            sys.exit(1)
        stored_task = parse_plan_task()
        print(f"\n{BOLD}Current task:{RESET}       {args.task}")
        print(f"{BOLD}Stored plan task:{RESET}  {stored_task or '(no frontmatter)'}")
        if stored_task and stored_task != args.task:
            if args.force_use_plan:
                print(f"{YELLOW}  Reuse allowed: yes (--force-use-plan overrides mismatch){RESET}")
                use_existing_plan = True
            else:
                print(f"{RED}  Reuse allowed: no (task mismatch){RESET}")
                print(f"{RED}Error: --use-plan requires the stored plan task to match --task.{RESET}")
                print(f"       Use --force-use-plan to override.{RESET}")
                sys.exit(1)
        else:
            print(f"{GREEN}  Reuse allowed: yes{RESET}")
            use_existing_plan = True
    else:
        if os.path.exists(PLAN_FILE) and os.path.getsize(PLAN_FILE) > 0:
            stored_task = parse_plan_task()
            print(f"\n{BOLD}Existing plan found in '{PLAN_FILE}':{RESET}")
            print(f"  {BOLD}Current task:{RESET}       {args.task}")
            print(f"  {BOLD}Stored plan task:{RESET}  {stored_task or '(no frontmatter)'}")
            if stored_task and stored_task != args.task:
                print(f"  {RED}Reuse allowed: no (task mismatch){RESET}")
                print(f"  {YELLOW}Auto-removing incompatible plan.{RESET}")
                os.remove(PLAN_FILE)
            else:
                print(f"  {GREEN}Reuse allowed: yes{RESET}")
                if not args.no_interactive:
                    choice = input(f"\n{BOLD}{YELLOW}Reuse existing plan? (y/n): {RESET}").strip().lower()
                    if choice == 'y':
                        use_existing_plan = True
                    else:
                        os.remove(PLAN_FILE)
                else:
                    use_existing_plan = True

    log_session(f"New multiswarm task initiated: {args.task}")

    # Phase 0 (optional): read-only recon by agy to seed the implementer. Non-blocking.
    if args.recon:
        run_recon(args.task, args.model_agy, args.skip_permissions, args.recon_timeout)

    success = False
    for iteration in range(1, args.iterations + 1):
        print(f"\n{BOLD}{YELLOW}>>> STARTING ITERATION {iteration} of {args.iterations} <<<{RESET}")

        # Read latest critique if present
        critique = None
        if iteration > 1 and os.path.exists(CRITIQUE_FILE):
            with open(CRITIQUE_FILE, "r") as f:
                critique = f.read()

        # Step 1: Planning
        if use_existing_plan and iteration == 1:
            print(f"\n{BOLD}{CYAN}=== Phase 1: Planning (Reusing Existing Plan) ==={RESET}")
            print(f"Reusing existing plan from '{PLAN_FILE}'")
        else:
            if not run_planning(args.task, iteration, critique, args.model_agy, args.skip_permissions, args.continue_session,
                                 args.agy_print_timeout):
                print(f"{RED}Planning failed in iteration {iteration}. Aborting.{RESET}")
                break

        if not args.no_interactive:
            if use_existing_plan and iteration == 1:
                # Ask to proceed using the existing plan
                choice = input(f"\n{BOLD}{YELLOW}Proceed to Implementation with existing plan? (y/n): {RESET}").strip().lower()
            else:
                choice = input(f"\n{BOLD}{YELLOW}Plan updated in '{PLAN_FILE}'. Proceed to Implementation? (y/n): {RESET}").strip().lower()
            if choice != 'y':
                print("Swarm aborted by user.")
                break

        # Step 2: Implementation
        impl_model = args.model_implementer or (args.model_claude if args.implementer == "claude" else None)
        if not run_implementation(args.task, iteration, impl_model, args.skip_permissions, args.continue_session, args.implementer,
                                   args.agy_print_timeout, args.impl_timeout or None):
            print(f"{RED}Implementation failed in iteration {iteration}. Aborting.{RESET}")
            break

        if not args.no_interactive:
            choice = input(f"\n{BOLD}{YELLOW}Implementation completed. Proceed to Verification? (y/n): {RESET}").strip().lower()
            if choice != 'y':
                print("Swarm aborted by user.")
                break

        # Step 3: Local validation
        val_passed, val_log = run_validation()

        # Step 4: Critique and review
        if not run_critique(args.task, iteration, val_passed, val_log, args.model_opencode, args.agent_opencode, args.continue_session):
            print(f"{RED}Critique phase failed in iteration {iteration}. Aborting.{RESET}")
            break

        # Step 5: Check completion status
        if check_success():
            print(f"\n{BOLD}{GREEN}✔ Success! Verifier (opencode) has approved the changes.{RESET}")
            success = True
            break
        else:
            print(f"\n{BOLD}{YELLOW}⚠ Iteration {iteration} complete. Verification did not yield success.{RESET}")
            if os.path.exists(CRITIQUE_FILE):
                print(f"{BOLD}Critique:{RESET}")
                with open(CRITIQUE_FILE, "r") as f:
                    print(f.read())

            if iteration < args.iterations:
                if not args.no_interactive:
                    choice = input(f"\n{BOLD}{YELLOW}Proceed to Iteration {iteration + 1}? (y/n): {RESET}").strip().lower()
                    if choice != 'y':
                        print("Swarm aborted by user.")
                        break
            else:
                print(f"{RED}Reached max iterations ({args.iterations}) without achieving verified SUCCESS.{RESET}")

    # Cleanup temp files if requested
    if args.cleanup:
        print("\nCleaning up temporary multiswarm files...")
        for f in [PLAN_FILE, SUMMARY_FILE, CRITIQUE_FILE, VALIDATION_LOG]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleanup done.")

    if success:
        print(f"\n{BOLD}{GREEN}================================================================{RESET}")
        print(f"{BOLD}{GREEN}             MULTISWARM TASK COMPLETED SUCCESSFULLY             {RESET}")
        print(f"{BOLD}{GREEN}================================================================{RESET}")
        sys.exit(0)
    else:
        print(f"\n{BOLD}{RED}================================================================{RESET}")
        print(f"{BOLD}{RED}                MULTISWARM TASK FAILED OR ABORTED               {RESET}")
        print(f"{BOLD}{RED}================================================================{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
