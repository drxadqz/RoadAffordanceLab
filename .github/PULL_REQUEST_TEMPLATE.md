## Summary

- What changed?
- What representation, evaluation, or reproducibility problem does it address?

## Evidence

- [ ] Public release audit passes.
- [ ] New/changed result claims have a frozen config and checkpoint SHA.
- [ ] Top-1, Macro-F1, weakest-class F1, and protocol are reported.
- [ ] No raw dataset, local path, credential, or private checkpoint was added.

## Reproduction

```bash
python scripts/build_public_assets.py
python scripts/verify_public_release.py
```

## Claim boundary

Describe whether the evidence is train, validation, development proxy-test, or final test. Do not promote cross-protocol comparisons as SOTA.
