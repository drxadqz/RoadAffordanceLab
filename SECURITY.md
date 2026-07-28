# Security Policy

Please do not open a public issue for credentials, private dataset paths, or a vulnerability that could expose local files. Use GitHub's private vulnerability reporting for this repository when available.

The repository intentionally excludes API keys, `.env` files, raw datasets, and machine-local path files. Git LFS checkpoints are model artifacts and should still be loaded only in a trusted Python/PyTorch environment.
