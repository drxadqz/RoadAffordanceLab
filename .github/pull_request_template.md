## What changed

<!-- Describe the smallest meaningful unit of change. -->

## Why

<!-- State the hypothesis, bug or reproducibility issue. -->

## Validation

- [ ] `python -m compileall -q src scripts`
- [ ] `python scripts/verify_release.py`
- [ ] `python scripts/generate_readme_assets.py --check`
- [ ] Data split, seed, initialization and training horizon are stated for any metric claim
- [ ] No formal-test result was used to tune the proposed change
- [ ] No private data, absolute local paths, secrets or unreviewed binary artifacts were added

## Evidence

<!-- Link machine-readable metrics, resolved config and checkpoint SHA when applicable. -->
