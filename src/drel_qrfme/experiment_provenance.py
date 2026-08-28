from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


PROVENANCE_SCHEMA_VERSION = 1
_SENSITIVE_ARGUMENT_FRAGMENTS = (
    "api-key",
    "api_key",
    "password",
    "secret",
    "token",
    "credential",
)


def _jsonable(value: Any) -> Any:
    """Convert audit metadata to deterministic JSON-safe primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def redact_command_argv(arguments: list[str] | tuple[str, ...]) -> list[str]:
    """Retain reproducible CLI structure without persisting obvious secrets."""

    redacted: list[str] = []
    redact_next = False
    for raw_argument in arguments:
        argument = str(raw_argument)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        key, separator, _value = argument.partition("=")
        lowered_key = key.lower()
        sensitive = any(
            fragment in lowered_key
            for fragment in _SENSITIVE_ARGUMENT_FRAGMENTS
        )
        if sensitive and separator:
            redacted.append(f"{key}=<redacted>")
            continue
        redacted.append(argument)
        if sensitive and argument.startswith("-"):
            redact_next = True
    return redacted


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_row_order_audit(loader: DataLoader) -> dict[str, Any]:
    """Hash effective dataset rows without touching the loader's sampler.

    The source manifest byte hash binds every source column.  This additional
    digest binds the *effective* path/label sequence after deterministic
    filtering or per-class subsampling.  Iterating a sampler here would risk
    consuming mutable RNG state, so the sampler is deliberately not opened.
    """

    dataset = loader.dataset
    frame = getattr(dataset, "df", None)
    if frame is None or not hasattr(frame, "columns"):
        return {
            "available": False,
            "rows": int(len(dataset)),
            "error": "dataset does not expose a dataframe-like df attribute",
        }

    columns = [str(column) for column in frame.columns]
    label_column = (
        "class_label_canonical"
        if "class_label_canonical" in columns
        else "class_label"
        if "class_label" in columns
        else None
    )
    if "image_path" not in columns or label_column is None:
        return {
            "available": False,
            "rows": int(len(dataset)),
            "error": "effective dataset rows require image_path and class label columns",
        }

    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    selected = frame[["image_path", label_column]]
    for image_path, class_label in selected.itertuples(index=False, name=None):
        row = [str(image_path), str(class_label)]
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        label = str(class_label)
        counts[label] = counts.get(label, 0) + 1

    return {
        "available": True,
        "rows": int(len(selected)),
        "ordered_columns": ["image_path", label_column],
        "ordered_rows_sha256": digest.hexdigest(),
        "class_counts": dict(sorted(counts.items())),
    }


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _sampler_contract(loader: DataLoader) -> dict[str, Any]:
    """Describe sampling determinism without iterating or reordering it."""

    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    contract: dict[str, Any] = {
        "loader_class": _qualified_class_name(loader),
        "sampler_class": _qualified_class_name(sampler) if sampler is not None else None,
        "batch_sampler_class": (
            _qualified_class_name(batch_sampler) if batch_sampler is not None else None
        ),
        "loader_batches": int(len(loader)),
        "batch_size": (
            int(loader.batch_size) if getattr(loader, "batch_size", None) is not None else None
        ),
        "drop_last": bool(getattr(loader, "drop_last", False)),
        "num_workers": int(getattr(loader, "num_workers", 0)),
    }
    generator = getattr(loader, "generator", None)
    if isinstance(generator, torch.Generator):
        contract["loader_generator_initial_seed"] = int(generator.initial_seed())

    allowed_attributes = (
        "seed",
        "epoch",
        "start_index",
        "num_samples",
        "replacement",
        "rank",
        "num_replicas",
        "global_samples_per_epoch",
        "class_second_groups",
        "start_batch",
        "num_batches",
        "batch_size",
    )
    for prefix, value in (("sampler", sampler), ("batch_sampler", batch_sampler)):
        if value is None:
            continue
        attributes: dict[str, Any] = {}
        for name in allowed_attributes:
            if not hasattr(value, name):
                continue
            item = getattr(value, name)
            if isinstance(item, (str, int, float, bool)) or item is None:
                attributes[name] = item
        if attributes:
            contract[f"{prefix}_attributes"] = attributes
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _split_audit(
    *,
    split: str,
    loader: DataLoader,
    manifest_path: str | Path,
    configured_sha256: Any,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    result: dict[str, Any] = {
        "split": str(split),
        "manifest_path": str(path),
        "configured_manifest_sha256": (
            str(configured_sha256).strip().lower()
            if configured_sha256 is not None
            else None
        ),
        "manifest_exists": path.is_file(),
    }
    if path.is_file():
        actual = file_sha256(path)
        result["actual_manifest_sha256"] = actual
        declared = result["configured_manifest_sha256"]
        result["configured_matches_actual"] = declared is None or declared == actual
    else:
        result["actual_manifest_sha256"] = None
        result["configured_matches_actual"] = False
    result["effective_dataset"] = _effective_row_order_audit(loader)
    result["sampling"] = _sampler_contract(loader)
    return result


def _run_git(repo_root: Path, arguments: list[str]) -> dict[str, Any]:
    command = ["git", "-C", str(repo_root), *arguments]
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "returncode": None,
        }
    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": int(completed.returncode),
    }


def _source_tree_audit(repo_root: Path) -> dict[str, Any]:
    pathspec = [
        "src",
        "scripts",
        "configs",
        "train.py",
        "validate.py",
        "test.py",
        "pyproject.toml",
        "requirements.txt",
    ]
    listed = _run_git(
        repo_root,
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *pathspec],
    )
    if not listed["ok"]:
        return {
            "available": False,
            "error": listed["stderr"].strip() or "git ls-files failed",
            "returncode": listed["returncode"],
        }

    untracked = _run_git(
        repo_root,
        ["ls-files", "-z", "--others", "--exclude-standard", "--", *pathspec],
    )

    relative_paths = sorted(
        path
        for path in listed["stdout"].split("\0")
        if path
    )
    files: dict[str, str] = {}
    errors: list[str] = []
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            continue
        try:
            files[relative.replace("\\", "/")] = file_sha256(path)
        except OSError as exc:
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    return {
        "available": True,
        "file_count": len(files),
        "tree_sha256": canonical_sha256(files),
        "files": files,
        "untracked_files": (
            sorted(path for path in untracked["stdout"].split("\0") if path)
            if untracked["ok"]
            else []
        ),
        "untracked_error": (
            None
            if untracked["ok"]
            else untracked["stderr"].strip() or "git untracked-file query failed"
        ),
        "errors": errors,
    }


def git_audit(repo_root: str | Path) -> dict[str, Any]:
    """Return a fail-soft Git and current source-tree identity audit."""

    root = Path(repo_root).expanduser().resolve()
    commit = _run_git(root, ["rev-parse", "HEAD"])
    branch = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    # Suppress untracked discovery here to avoid traversing unrelated or
    # inaccessible experiment directories. Relevant untracked source files are
    # still included by _source_tree_audit below.
    status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=no"])
    source_tree = _source_tree_audit(root)
    tracked_status = [line for line in status["stdout"].splitlines() if line]
    tracked_dirty = bool(tracked_status)
    untracked_source_files = source_tree.get("untracked_files", [])
    source_untracked = bool(untracked_source_files)

    if not commit["ok"]:
        return {
            "available": False,
            "repo_root": str(root),
            "error": commit["stderr"].strip() or "git rev-parse failed",
            "source_tree": source_tree,
        }
    return {
        "available": True,
        "repo_root": str(root),
        "commit": commit["stdout"].strip(),
        "branch": branch["stdout"].strip() if branch["ok"] else None,
        "tracked_dirty": tracked_dirty,
        "source_untracked": source_untracked,
        "tracked_status": tracked_status,
        # A source-tree digest binds both tracked and relevant untracked files.
        # Treat the run as dirty when tracked files changed. The source tree is
        # authoritative for uncommitted/untracked training code identity.
        "dirty": tracked_dirty or source_untracked,
        "status_error": status["stderr"].strip() if not status["ok"] else None,
        "source_tree": source_tree,
    }


def model_audit(model: nn.Module) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    schema = {
        "parameters": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in parameters
        ],
        "buffers": [
            {
                "name": name,
                "shape": list(buffer.shape),
                "dtype": str(buffer.dtype),
            }
            for name, buffer in buffers
        ],
    }
    backbone = getattr(model, "backbone", None)
    total_parameters = sum(parameter.numel() for _, parameter in parameters)
    trainable_parameters = sum(
        parameter.numel()
        for _, parameter in parameters
        if parameter.requires_grad
    )
    total_bytes = sum(
        parameter.numel() * parameter.element_size()
        for _, parameter in parameters
    )
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for _, parameter in parameters
        if parameter.requires_grad
    )
    return {
        "model_class": _qualified_class_name(model),
        "backbone_class": _qualified_class_name(backbone) if backbone is not None else None,
        "parameter_tensors": len(parameters),
        "trainable_parameter_tensors": sum(
            int(parameter.requires_grad) for _, parameter in parameters
        ),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "total_parameter_bytes": int(total_bytes),
        "trainable_parameter_bytes": int(trainable_bytes),
        "buffer_tensors": len(buffers),
        "parameter_schema_sha256": canonical_sha256(schema),
    }


def build_run_provenance(
    *,
    config_path: str | Path,
    resolved_config_path: str | Path,
    repo_root: str | Path,
    cfg: Mapping[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    runtime_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build immutable, side-effect-free training-run provenance metadata."""

    source_config = Path(config_path).expanduser().resolve()
    resolved_config = Path(resolved_config_path).expanduser().resolve()
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise TypeError("cfg.data must be a mapping for provenance")

    datasets = {
        "train": _split_audit(
            split="train",
            loader=train_loader,
            manifest_path=data_cfg["train_manifest"],
            configured_sha256=data_cfg.get("train_manifest_sha256"),
        ),
        "val": _split_audit(
            split="val",
            loader=val_loader,
            manifest_path=data_cfg["val_manifest"],
            configured_sha256=data_cfg.get("val_manifest_sha256"),
        ),
    }
    if test_loader is not None:
        datasets["test"] = _split_audit(
            split="test",
            loader=test_loader,
            manifest_path=data_cfg["test_manifest"],
            configured_sha256=data_cfg.get("test_manifest_sha256"),
        )

    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(cfg.get("seed", 79)),
        "command": {
            "executable": str(Path(sys.executable).resolve()),
            "argv": redact_command_argv(tuple(str(item) for item in sys.argv)),
            "cwd": str(Path.cwd().resolve()),
        },
        "config": {
            "source_path": str(source_config),
            "source_sha256": file_sha256(source_config),
            "resolved_path": str(resolved_config),
            "resolved_sha256": file_sha256(resolved_config),
        },
        "git": git_audit(repo_root),
        "datasets": datasets,
        "model": model_audit(model),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda) if torch.version.cuda is not None else None,
            "runtime_audit": _jsonable(runtime_audit or {}),
        },
    }
    payload["provenance_sha256"] = canonical_sha256(payload)
    return payload


def write_run_provenance(provenance: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(provenance), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "build_run_provenance",
    "canonical_sha256",
    "file_sha256",
    "git_audit",
    "model_audit",
    "redact_command_argv",
    "write_run_provenance",
]
