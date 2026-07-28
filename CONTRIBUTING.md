# Contributing to RoadAffordanceLab

Thank you for improving the project. Contributions are welcome when they preserve the distinction between a research hypothesis, a validation result, and a frozen test claim.

## Development workflow

1. Create a focused branch from `main`.
2. Keep one conceptual change per pull request.
3. Add or update a public YAML config when behavior changes.
4. Run the repository contract and Python compilation checks.
5. Describe the exact data split, checkpoint SHA, seed, and evaluation protocol for metric changes.

```bash
python scripts/check_repository_contract.py
python -m compileall -q src scripts/check_repository_contract.py
```

## Experimental claims

- Use validation data for model and checkpoint selection.
- Do not tune architecture, thresholds, or losses against the formal test split.
- Label development-subset results as development evidence.
- Report Top-1 and Macro-F1 together with weakest-class or bottom-class behavior.
- State whether pretraining, TTA, ensembles, teachers, or class-specific corrections are used.
- Link every promoted result to a machine-readable metrics payload and checkpoint SHA.

## Public-release safety

Before opening a pull request, remove:

- raw dataset images that cannot be redistributed;
- API keys, tokens, passwords, proxy credentials, and private URLs;
- machine-specific absolute paths and user names;
- large generated artifacts that are not intentionally managed by Git LFS.

Use the issue template for reproducible bugs. For research proposals, include the mechanism being tested, a matched control, a stopping rule, and the evidence that would falsify the proposal.

