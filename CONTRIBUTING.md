# Contributing

Contributions that improve reproducibility, documentation, testing or scientifically controlled comparisons are welcome.

## Before opening a pull request

1. Create a focused branch from `main`.
2. Keep generated datasets, local manifests, checkpoints and secrets out of Git.
3. Add or update an experiment config when behavior changes.
4. State the exact data split, seed, initialization and training horizon for metric claims.
5. Do not use the formal test split to select a model or tune a mechanism.
6. Run the release checks:

```bash
python -m compileall -q src scripts
python scripts/verify_release.py
python scripts/generate_readme_assets.py --check
```

## Result contributions

A result table should link to machine-readable evidence. At minimum include:

- resolved configuration;
- checkpoint SHA256;
- Top-1 and Macro-F1;
- per-class metrics;
- split size and sample-selection rules;
- whether pretraining, teacher models, TTA or ensembles were used.

Comparisons must be matched on data, input geometry, pretraining, training horizon and inference policy. Negative results are useful when the hypothesis and stopping rule were registered before evaluation.

## Code style

- Target Python 3.11.
- Prefer typed, explicit interfaces.
- Keep filesystem paths configurable.
- Keep stdout machine-readable where a script is consumed by automation; send diagnostics to stderr.
- Avoid silent checkpoint key drops.
- Preserve existing checkpoint compatibility unless a migration is documented.

## Pull request scope

Small, reviewable pull requests are preferred. Separate model changes, data-protocol changes and result-reporting changes when possible so that causal attribution remains clear.
