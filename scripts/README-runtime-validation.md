# Runtime Validation

Generation health must be checked with a known-good GGUF, not with an arbitrary
local model. A bad or misconverted model can fail generation while the runtime is
healthy.

Use `scripts/validate-runtime.sh` after building `llama-server`:

```bash
KNOWN_GOOD_GEN_MODEL=/path/to/known-good.gguf scripts/validate-runtime.sh
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

The current local failure pattern was isolated this way: the BF16 candidate failed
the cold generation smoke, while a trusted Qwen2.5-Coder GGUF passed with the same
server binary.
