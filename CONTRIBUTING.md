# Contributing

Contributions are welcome when they improve reproducibility, correctness, or mechanism-level understanding.

## Before opening a pull request

1. Keep datasets, local absolute paths, API keys, and private checkpoints out of Git.
2. Put machine-specific paths in `configs/data/local_paths.yaml` (ignored by Git).
3. Add or update a portable config for any behavior change.
4. Report Top-1, Macro-F1, weakest-class F1, and the evaluation protocol.
5. Distinguish validation evidence from final-test evidence.
6. Rebuild and verify public assets:

```bash
python scripts/build_public_assets.py
python scripts/verify_public_release.py
```

## Good issue reports

Please include:

- operating system, Python, PyTorch, CUDA, and GPU versions;
- the exact command and config path;
- the full error traceback;
- whether Git LFS checkpoints were downloaded;
- a minimal reproduction that does not require private data.

## Result changes

Do not edit SVG metrics by hand. Update the source JSON/CSV result payloads only when they come from a frozen, auditable run, then regenerate the figures. A new headline result also needs a resolved config, checkpoint SHA256, and protocol note.
