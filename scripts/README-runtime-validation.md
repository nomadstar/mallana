# Runtime Validation

Generation health must be checked with a known-good GGUF, not with an arbitrary
local model. A bad or misconverted model can fail generation while the runtime is
healthy.

**Choose a ROBUST instruct model as the known-good reference** — e.g.
`qwen2.5-3b-instruct-q4_k_m.gguf`. Do **not** use a `coder-1.5b` (especially the BF16
build): it is fp-fragile and collapses into repetition loops under any slightly-lossy KV,
so it fails as a "known-good" and mislabels a perfectly healthy runtime as broken. The 3B
instruct model runs every KV config (f16, q8_0, turbo2/3) coherently on CPU and GPU.

Use `scripts/validate-runtime.sh` after building `llama-server`:

```bash
KNOWN_GOOD_GEN_MODEL=~/models/qwen2.5-3b-instruct-q4_k_m.gguf scripts/validate-runtime.sh
```

To classify another model without treating it as proof that the runtime is broken:

```bash
KNOWN_GOOD_GEN_MODEL=/path/to/known-good.gguf \
MODEL=/path/to/candidate.gguf \
scripts/validate-runtime.sh
```

Interpretation:

- Known-good passes: the built runtime can generate coherent text.
- Candidate fails while known-good passes: the candidate model is suspect.
- Known-good fails: the runtime, build, server settings, or smoke test are suspect.

A past failure was isolated this way: a BF16 candidate failed the cold generation smoke
while a Q4_K_M reference passed with the same server binary — proving the runtime was
healthy and the candidate model was the problem. (Later confirmed at larger scale: the
whole "turbo corruption" saga traced back to a weak coder-1.5b test model, not the
kernels — always swap in the real target model before blaming the runtime.)
