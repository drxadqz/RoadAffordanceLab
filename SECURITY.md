# Security Policy

## Supported version

Security fixes are applied to the latest `main` branch.

## Reporting a vulnerability

Please do not disclose credentials, private dataset paths, tokens or exploitable issues in a public issue. Use GitHub's private security advisory workflow for this repository.

Include:

- affected commit or file;
- minimal reproduction steps;
- potential impact;
- whether the report contains private data or credentials.

## Sensitive artifacts

This repository must not contain:

- API keys, passwords or proxy credentials;
- private dataset images or manifests with personal absolute paths;
- unreviewed executable checkpoints from unknown sources;
- unrelated user or system logs.

PyTorch checkpoint files can execute Python through unsafe pickle payloads. Only load the provided artifacts when their Git-LFS OID and SHA256 agree with [`results/s7_lineage/checkpoint_manifest.json`](results/s7_lineage/checkpoint_manifest.json), and do not load untrusted checkpoints.
