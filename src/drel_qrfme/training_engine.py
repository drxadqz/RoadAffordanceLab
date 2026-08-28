from __future__ import annotations

# =============================================================================
# 训练引擎阅读导航（正式 DREL-QRFME 实验只需重点读这些位置）
# =============================================================================
# 1) load_config               ：加载 base YAML，并用正式 YAML 覆盖；
# 2) RSCDSurfaceDataset        ：从冻结 manifest 读取图片与 27 类标签；
# 3) build_loaders             ：建立训练/验证 DataLoader；
# 4) build_model               ：正式配置只走 drel_rt/rfme 分支；
# 5) balanced_softmax_training_logits：仅训练时加入 log(class_count)；
# 6) train_one_epoch           ：选择训练路径；正式配置进入通用 CE 路径；
# 7) evaluate                  ：始终使用模型原始 logits 计算验证指标；
# 8) run_train                ：恢复断点、创建优化器、循环 epoch、保存结果。
#
# 本文件较长，是因为它保留了早期消融实验所需的兼容 loss/训练分支。正式配置中
# 这些分支的权重均为 0；不要为了“看起来简短”删除它们，否则可能改变配置检查、
# 日志字段或旧 checkpoint 的恢复行为。阅读当前算法时按上面 8 个入口即可。
# =============================================================================

import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
import yaml

"""DREL-QRFME 的训练、验证、断点恢复和指标计算主引擎。

普通用户不需要直接修改本文件。实验参数统一放在 ``configs/`` 中；
跨电脑路径由 ``scripts/run_training.py`` 在运行时映射，避免手改硬编码路径。
本文件保留了经过 49 个完整 epoch 实际验证的训练逻辑，因此不要为了“精简”
随意改动优化器恢复、随机状态恢复、manifest 哈希或 checkpoint 校验代码。
"""

from drel_qrfme.training_losses import (
    c3_total_aux_loss,
    ceta_calibration_loss,
    factor_graph_metric_loss,
    mechanism_routed_tournament_loss,
)
from drel_qrfme.models.drel_qrfme_model import DRELRTSurfaceClassifier
from drel_qrfme.training_monitor import TrainingMonitor


def _runtime_path(path: str | Path) -> Path:
    """Resolve author-independent package and dataset URIs.

    ``release://`` points inside this package. ``dataset://`` points inside
    the RSCD root supplied to the portable launcher.
    """

    text = str(path)
    normalized = text.replace("\\", "/")
    if normalized.startswith("release://"):
        release_root = Path(__file__).resolve().parents[2]
        return release_root / Path(normalized.removeprefix("release://"))
    if normalized.startswith("dataset://"):
        dataset_root = os.environ.get("DREL_DATASET_ROOT", "").strip()
        if not dataset_root:
            raise RuntimeError(
                "dataset:// path requires DREL_DATASET_ROOT; launch with "
                "scripts/run_training.py --dataset-root ..."
            )
        return Path(dataset_root) / Path(normalized.removeprefix("dataset://"))
    return Path(text)
from drel_qrfme.experiment_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    build_run_provenance,
    canonical_sha256,
    write_run_provenance,
)
from drel_qrfme.rscd_label_factors import (
    FACTOR_AXES,
    FACTOR_LABELS,
    RSCDFactorSpec,
    build_rscd_factor_spec,
    canonical_class_label,
    fixed_rscd_class_map,
    parse_rscd_label,
    resolve_rscd_class_label,
    sanity_summary,
)
from drel_qrfme.rscd_epoch_sampler import (
    EpochClassSecondRotationSampler,
)
from drel_qrfme.training_helpers import (
    ModelEMA,
    coral_roughness_loss,
    set_warmup_cosine_lr,
    weighted_validation_score,
)
from drel_qrfme.image_transforms import (
    build_transforms,
    build_transforms_with_valid_mask,
)
from drel_qrfme.runtime_utils import resolve_device, set_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _capture_rng_state() -> dict[str, Any]:
    """保存 Python/NumPy/PyTorch 随机状态，保证步级断点可精确续训。"""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    """恢复所有随机数发生器；必须在继续取样和数据增强之前调用。"""
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _nonfinite_gradient_recovery_settings(
    train_cfg: Mapping[str, Any],
) -> tuple[bool, int]:
    enabled = bool(
        train_cfg.get("skip_nonfinite_grad_steps", False)
    ) or _env_flag_enabled("DREL_SKIP_NONFINITE_GRAD_STEPS")
    default_max = 32 if enabled else 0
    raw_max = train_cfg.get(
        "max_nonfinite_grad_skips",
        os.environ.get("DREL_MAX_NONFINITE_GRAD_SKIPS", default_max),
    )
    max_skips = int(raw_max or 0)
    if max_skips < 0:
        raise RuntimeError("max_nonfinite_grad_skips cannot be negative")
    return enabled, max_skips


def _nonfinite_gradient_diagnostics(
    model: nn.Module,
    batch: Mapping[str, Any],
    label: torch.Tensor,
    idx_to_class: Mapping[int, str],
) -> dict[str, Any]:
    """Return compact diagnostics for a skipped non-finite gradient step."""

    param_preview: list[dict[str, Any]] = []
    nonfinite_param_count = 0
    nonfinite_element_count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all()):
                continue
            nonfinite_param_count += 1
            finite_count = int(finite_mask.sum().detach().cpu())
            total_count = int(grad.numel())
            bad_count = total_count - finite_count
            nonfinite_element_count += bad_count
            if len(param_preview) < 20:
                finite_abs_max = None
                if finite_count > 0:
                    finite_abs_max = float(
                        grad.detach()[finite_mask].abs().max().cpu()
                    )
                param_preview.append(
                    {
                        "name": str(name),
                        "shape": [int(dim) for dim in grad.shape],
                        "dtype": str(grad.dtype),
                        "nonfinite": int(bad_count),
                        "nan": int(torch.isnan(grad).sum().detach().cpu()),
                        "posinf": int(torch.isposinf(grad).sum().detach().cpu()),
                        "neginf": int(torch.isneginf(grad).sum().detach().cpu()),
                        "finite_abs_max": finite_abs_max,
                    }
                )

    label_counts: dict[str, int] = {}
    for value in label.detach().cpu().tolist():
        name = str(idx_to_class.get(int(value), int(value)))
        label_counts[name] = label_counts.get(name, 0) + 1

    paths = [str(path) for path in batch.get("image_path", [])]
    event: dict[str, Any] = {
        "nonfinite_param_count": int(nonfinite_param_count),
        "nonfinite_element_count": int(nonfinite_element_count),
        "nonfinite_param_preview": param_preview,
        "label_counts": label_counts,
    }
    if paths:
        event["image_path_count"] = len(paths)
        event["image_path_sha256"] = hashlib.sha256(
            "\n".join(paths).encode("utf-8")
        ).hexdigest()
        event["image_path_preview"] = paths[:8]
    return event


def _deep_update_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_update_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_file(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"config extends cycle detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop("extends", None)
    if extends:
        base_path = Path(str(extends))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_cfg = _load_config_file(base_path, seen)
        cfg = _deep_update_config(base_cfg, cfg)
    return cfg


def load_config(path: Path) -> dict[str, Any]:
    """递归合并 YAML 配置并完成兼容字段规范化。

    正式配置 ``rscd_full_train_seed097.yaml`` 通过 ``extends`` 继承基础模型。
    后出现的键覆盖基础键；字典递归合并，而不是整段替换。
    """
    cfg = _load_config_file(path)
    cfg.setdefault("seed", 79)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("output_dir", "outputs/c3_farnet")
    _normalize_model_config(cfg["model"])
    _normalize_loss_config(cfg["loss"])
    _validate_arcq_composition_consistency_config(cfg)
    return cfg


def _amp_autocast_dtype(
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.dtype | None:
    if device.type != "cuda" or not bool(train_cfg.get("amp", True)):
        return None
    requested = str(train_cfg.get("amp_dtype", "auto")).strip().lower()
    if requested in {"bf16", "bfloat16", "auto"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if requested != "auto":
            print("WARNING: CUDA bf16 is unavailable; falling back to fp16 autocast")
    if requested not in {"auto", "bf16", "bfloat16", "fp16", "float16"}:
        raise ValueError(f"unsupported train.amp_dtype: {requested}")
    return torch.float16


def _eval_autocast_dtype(
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.dtype | None:
    """Resolve validation autocast independently from gradient scaling.

    Evaluation previously ran entirely in fp32 even when the matching training
    path used bf16.  Fixed-size RSCD validation is a material fraction of every
    hour-scale screen, so allow the same numerically safe CUDA autocast dtype
    while keeping metric accumulation in fp32.  It is opt-in through
    ``train.eval_amp=true`` so historical validation remains fp32 by default.
    """

    if device.type != "cuda":
        return None
    # Preserve the historical evaluation contract unless a research config
    # opts in explicitly.  A low-precision forward can change argmax and hence
    # checkpoint selection even though logits are accumulated in fp32.
    if not bool(train_cfg.get("eval_amp", False)):
        return None
    amp_cfg = dict(train_cfg)
    amp_cfg["amp"] = True
    return _amp_autocast_dtype(amp_cfg, device)


def _configure_cuda_runtime(
    train_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Apply explicit, auditable CUDA throughput options for fixed-size runs."""

    audit = {
        "device": device.type,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
        "deterministic_warn_only": False,
        "allow_tf32": False,
        "float32_matmul_precision": None,
    }
    deterministic_algorithms = bool(
        train_cfg.get("deterministic_algorithms", False)
    )
    deterministic_warn_only = bool(
        train_cfg.get("deterministic_warn_only", False)
    )
    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=deterministic_warn_only,
    )
    audit.update(
        {
            "deterministic_algorithms": deterministic_algorithms,
            "deterministic_warn_only": deterministic_warn_only,
        }
    )
    if device.type != "cuda":
        return audit

    cudnn_benchmark = bool(train_cfg.get("cudnn_benchmark", False))
    cudnn_deterministic = bool(train_cfg.get("cudnn_deterministic", False))
    allow_tf32 = bool(train_cfg.get("allow_tf32", False))
    if cudnn_benchmark and cudnn_deterministic:
        raise ValueError(
            "train.cudnn_benchmark and train.cudnn_deterministic cannot both "
            "be true: benchmark mode may select different convolution "
            "algorithms across otherwise matched evaluation processes"
        )
    precision = str(
        train_cfg.get(
            "float32_matmul_precision",
            "high" if allow_tf32 else "highest",
        )
    ).strip().lower()
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "train.float32_matmul_precision must be highest/high/medium, "
            f"got {precision!r}"
        )

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision(precision)
    audit.update(
        {
            "cudnn_benchmark": cudnn_benchmark,
            "cudnn_deterministic": cudnn_deterministic,
            "deterministic_algorithms": deterministic_algorithms,
            "deterministic_warn_only": deterministic_warn_only,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "allow_tf32": allow_tf32,
            "float32_matmul_precision": precision,
        }
    )
    return audit


def _balanced_sampling_active(train_cfg: dict[str, Any], epoch: int) -> bool:
    if bool(train_cfg.get("temporal_rotation_sampling", False)):
        # One-per-class-second rotation is itself the epoch sampling protocol.
        # Replacement-based balancing would violate its uniqueness guarantee.
        return False
    delayed_start = int(train_cfg.get("delayed_balanced_sampling_start_epoch", 0) or 0)
    if delayed_start > 0:
        return int(epoch) >= delayed_start
    return bool(train_cfg.get("balanced_sampling", True))


def _accumulation_window(
    step: int,
    *,
    total_steps: int,
    accumulation_steps: int,
) -> tuple[int, bool]:
    """Return the current micro-batch window size and update boundary.

    The final window can contain fewer than ``accumulation_steps`` batches. Its
    loss must be divided by that actual count; otherwise the last optimizer
    update of every non-divisible epoch is systematically under-scaled.
    """

    step = int(step)
    total_steps = int(total_steps)
    accumulation_steps = max(int(accumulation_steps), 1)
    if step <= 0 or total_steps <= 0 or step > total_steps:
        raise ValueError(
            f"invalid accumulation position: step={step} total_steps={total_steps}"
        )
    window_start = ((step - 1) // accumulation_steps) * accumulation_steps + 1
    window_end = min(
        ((step - 1) // accumulation_steps + 1) * accumulation_steps,
        total_steps,
    )
    return window_end - window_start + 1, step == window_end


def _completed_optimizer_updates_before_step(
    *,
    epoch: int,
    step: int,
    total_steps: int,
    accumulation_steps: int,
) -> int:
    """Count completed optimizer updates before a 1-based micro-batch step.

    This definition includes a possibly short final accumulation window and is
    stable under atomic mid-epoch resume because ``step`` and ``total_steps``
    retain their original epoch coordinates.
    """

    epoch = int(epoch)
    step = int(step)
    total_steps = int(total_steps)
    accumulation_steps = max(int(accumulation_steps), 1)
    if epoch <= 0 or step <= 0 or step > total_steps:
        raise ValueError(
            "invalid optimizer-update position: "
            f"epoch={epoch} step={step} total_steps={total_steps}"
        )
    updates_per_epoch = math.ceil(total_steps / accumulation_steps)
    return (
        (epoch - 1) * updates_per_epoch
        + (step - 1) // accumulation_steps
    )


def _set_optimizer_update_curricula(
    model: nn.Module,
    completed_updates: int,
) -> None:
    """Apply optimizer-update curricula without coupling generic training code.

    Current experiments expose the setter on the backbone.  The module walk is
    intentional: it remains correct for ordinary classifiers, DDP-style
    wrappers and future wrappers, while doing nothing for every existing model.
    """

    setters = []
    for module in model.modules():
        setter = getattr(module, "set_rcrc_optimizer_step", None)
        if callable(setter):
            setters.append(setter)
    if len(setters) > 1:
        raise RuntimeError("multiple RCRC curriculum owners found in one model")
    if setters:
        setters[0](int(completed_updates))


def _optimizer_update_curriculum_logs(model: nn.Module) -> dict[str, float]:
    """Return small, history-safe diagnostics for active curricula."""

    states = []
    for module in model.modules():
        getter = getattr(module, "rcrc_schedule_state", None)
        if callable(getter):
            states.append(getter())
    if len(states) > 1:
        raise RuntimeError("multiple RCRC curriculum states found in one model")
    if not states:
        return {}
    state = states[0]
    return {
        "rcrc_completed_updates": float(state["completed_updates"]),
        "rcrc_start_update": float(state["start_update"]),
        "rcrc_end_update": float(state["end_update"]),
        "rcrc_lambda": float(state["lambda"]),
    }


def _normalize_model_config(model_cfg: dict[str, Any]) -> None:
    """Accept whitepaper-style config aliases while keeping old configs valid."""

    head_cfg = model_cfg.get("head", {}) if isinstance(model_cfg.get("head"), dict) else {}
    if "head_type" not in model_cfg and "type" in head_cfg:
        model_cfg["head_type"] = head_cfg["type"]
    if "use_dry_vor" in model_cfg and "use_dry_concrete_roughness_vor_residual" not in model_cfg:
        model_cfg["use_dry_concrete_roughness_vor_residual"] = bool(model_cfg["use_dry_vor"])


def _normalize_loss_config(loss_cfg: dict[str, Any]) -> None:
    """Map C3-FaRNet paper notation to the implementation's loss weights."""

    axis_weights = dict(loss_cfg.get("factor_axis_weights", {}) or {})
    alias_to_axis = {
        "lambda_factor_f": "friction",
        "lambda_factor_m": "material",
        "lambda_factor_r": "roughness",
    }
    for alias, axis in alias_to_axis.items():
        if alias in loss_cfg:
            axis_weights[axis] = float(loss_cfg[alias])
    if axis_weights:
        loss_cfg["factor_axis_weights"] = {
            "friction": float(axis_weights.get("friction", 1.0)),
            "material": float(axis_weights.get("material", 1.0)),
            "roughness": float(axis_weights.get("roughness", 1.0)),
        }

    if "lambda_factor" in loss_cfg and "factor_weight" not in loss_cfg:
        loss_cfg["factor_weight"] = float(loss_cfg["lambda_factor"])
    if "lambda_tournament" in loss_cfg and "tournament_weight" not in loss_cfg:
        loss_cfg["tournament_weight"] = float(loss_cfg["lambda_tournament"])
    if "lambda_counterfactual" in loss_cfg and "counterfactual_weight" not in loss_cfg:
        loss_cfg["counterfactual_weight"] = float(loss_cfg["lambda_counterfactual"])
    if "lambda_reliability" in loss_cfg and "reliability_weight" not in loss_cfg:
        loss_cfg["reliability_weight"] = float(loss_cfg["lambda_reliability"])

    use_to_weight = {
        "use_factor_ce": ("factor_weight", 0.3),
        "use_tournament": ("tournament_weight", 0.1),
        "use_counterfactual": ("counterfactual_weight", 0.05),
        "use_reliability": ("reliability_weight", 0.05),
    }
    for flag, (weight_name, default_weight) in use_to_weight.items():
        if flag not in loss_cfg:
            continue
        if not bool(loss_cfg[flag]):
            loss_cfg[weight_name] = 0.0
        elif weight_name not in loss_cfg:
            loss_cfg[weight_name] = float(default_weight)


def _validate_arcq_composition_consistency_config(
    cfg: dict[str, Any],
) -> None:
    """Reject a consistency objective that would revive disabled C evidence.

    ARCQ energy-only controls keep the C parameters allocated solely for
    checkpoint/parameter-count matching.  A positive protected-composition
    consistency weight must therefore never make those dormant parameters
    active through a second weak-view forward.
    """

    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    if not isinstance(loss_cfg, dict) or not isinstance(model_cfg, dict):
        return
    weight = float(
        loss_cfg.get("arcq_composition_consistency_weight", 0.0) or 0.0
    )
    if weight <= 0.0:
        return
    backbone_kwargs = model_cfg.get("backbone_kwargs", {})
    if not isinstance(backbone_kwargs, dict):
        return
    if not bool(backbone_kwargs.get("use_composition_evidence", True)):
        raise ValueError(
            "loss.arcq_composition_consistency_weight > 0 requires "
            "model.backbone_kwargs.use_composition_evidence=true; a "
            "composition-free ablation cannot execute or optimize the "
            "disabled C path"
        )


def build_class_map(manifests: list[Path]) -> dict[str, int]:
    class_to_idx = fixed_rscd_class_map()
    expected = set(class_to_idx)
    observed: set[str] = set()
    recovered = 0
    for manifest in manifests:
        df = pd.read_csv(
            manifest,
            usecols=["class_label", "image_path"],
            dtype=str,
            low_memory=False,
        )
        resolved = [
            resolve_rscd_class_label(label, image_path)
            for label, image_path in zip(
                df["class_label"],
                df["image_path"],
                strict=True,
            )
        ]
        recovered += sum(
            canonical_class_label(label) != resolved_label
            and resolved_label in expected
            for label, resolved_label in zip(
                df["class_label"],
                resolved,
                strict=True,
            )
        )
        observed.update(resolved)
    if recovered:
        print(
            "WARNING: recovered "
            f"{recovered} malformed RSCD manifest label(s) from unambiguous "
            "filename class suffixes"
        )
    unknown = sorted(observed - expected)
    if unknown:
        print(
            "WARNING: manifests contain labels outside the fixed RSCD-27 closed set; "
            f"they will be ignored: {unknown}"
        )
    missing = sorted(expected - observed)
    if missing:
        print(f"WARNING: fixed RSCD-27 labels absent from the supplied manifests: {missing}")
    return class_to_idx


class RSCDSurfaceDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        *,
        class_to_idx: dict[str, int],
        transform,
        max_samples: int | None = None,
        max_samples_per_class: int | None = None,
        seed: int = 79,
        recover_unreadable: bool = True,
        emit_valid_mask: bool = False,
        secondary_transform=None,
        secondary_key: str = "secondary_image",
    ) -> None:
        df = pd.read_csv(manifest, dtype=str, low_memory=False)
        df["class_label_canonical"] = [
            resolve_rscd_class_label(label, image_path)
            for label, image_path in zip(
                df["class_label"],
                df["image_path"],
                strict=True,
            )
        ]
        df = df[df["class_label_canonical"].isin(class_to_idx)].copy()
        if max_samples_per_class:
            parts = []
            for _, group in df.groupby("class_label_canonical", sort=True):
                n = min(int(max_samples_per_class), len(group))
                parts.append(group.sample(n=n, random_state=int(seed)))
            df = pd.concat(parts, ignore_index=True)
        if max_samples:
            df = df.sample(n=min(int(max_samples), len(df)), random_state=int(seed)).reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.recover_unreadable = bool(recover_unreadable)
        self.emit_valid_mask = bool(emit_valid_mask)
        self.secondary_transform = secondary_transform
        self.secondary_key = str(secondary_key)
        if self.secondary_transform is not None and self.emit_valid_mask:
            raise ValueError(
                "secondary image views and explicit valid masks cannot be "
                "combined by the current dataset contract"
            )
        if self.secondary_transform is not None and not self.secondary_key:
            raise ValueError("secondary_key must be non-empty")
        self._warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = min(50, len(self.df)) if self.recover_unreadable else 1
        for offset in range(attempts):
            row = self.df.iloc[(int(idx) + offset) % len(self.df)]
            source_path = str(row["image_path"])
            path = _runtime_path(source_path)
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.load()
                    if self.secondary_transform is None:
                        transformed = self.transform(image)
                        secondary_tensor = None
                    else:
                        # Both views must describe the same augmented scene.
                        # Replay the primary transform's random draws for the
                        # native-resolution observer, then preserve the RNG
                        # state produced by the canonical primary transform.
                        python_state = random.getstate()
                        numpy_state = np.random.get_state()
                        torch_state = torch.get_rng_state()
                        transformed = self.transform(image.copy())
                        python_after = random.getstate()
                        numpy_after = np.random.get_state()
                        torch_after = torch.get_rng_state()
                        random.setstate(python_state)
                        np.random.set_state(numpy_state)
                        torch.set_rng_state(torch_state)
                        secondary_tensor = self.secondary_transform(image.copy())
                        random.setstate(python_after)
                        np.random.set_state(numpy_after)
                        torch.set_rng_state(torch_after)
                valid_mask: torch.Tensor | None = None
                if self.emit_valid_mask:
                    if not (
                        isinstance(transformed, tuple)
                        and len(transformed) == 2
                    ):
                        raise TypeError(
                            "emit_valid_mask=True requires a transform returning "
                            "(image_tensor, valid_mask)"
                        )
                    tensor, valid_mask = transformed
                    if not isinstance(tensor, torch.Tensor) or not isinstance(
                        valid_mask,
                        torch.Tensor,
                    ):
                        raise TypeError(
                            "paired image/mask transform must return two tensors"
                        )
                    if valid_mask.ndim != 3 or valid_mask.shape[0] != 1:
                        raise ValueError(
                            "valid_mask must have shape 1xHxW, got "
                            f"{tuple(valid_mask.shape)}"
                        )
                    if tuple(valid_mask.shape[-2:]) != tuple(tensor.shape[-2:]):
                        raise ValueError(
                            "image and valid_mask must be spatially aligned: "
                            f"image={tuple(tensor.shape)} "
                            f"mask={tuple(valid_mask.shape)}"
                        )
                else:
                    tensor = transformed
                label_name = canonical_class_label(row["class_label_canonical"])
                triple = parse_rscd_label(label_name)
                item = {
                    "image": tensor,
                    "label": torch.tensor(self.class_to_idx[label_name], dtype=torch.long),
                    "friction_factor": torch.tensor(triple.friction, dtype=torch.long),
                    "material_factor": torch.tensor(triple.material, dtype=torch.long),
                    "roughness_factor": torch.tensor(triple.roughness, dtype=torch.long),
                    "class_label": label_name,
                    "image_path": source_path,
                }
                if valid_mask is not None:
                    item["valid_mask"] = valid_mask
                if secondary_tensor is not None:
                    if not isinstance(secondary_tensor, torch.Tensor):
                        raise TypeError("secondary transform must return a tensor")
                    item[self.secondary_key] = secondary_tensor
                return item
            except (OSError, SyntaxError, ValueError) as exc:
                last_error = exc
                if not self.recover_unreadable:
                    raise RuntimeError(
                        "Evaluation image is unreadable; refusing to replace it "
                        f"with another sample because that would change the frozen "
                        f"protocol: index={int(idx)} path={path}"
                    ) from exc
                path_text = str(path)
                if path_text not in self._warned:
                    self._warned.add(path_text)
                    print(f"WARNING: skipped unreadable image: {path_text} ({type(exc).__name__}: {exc})")
        raise RuntimeError(f"Could not load image near index {idx}: {last_error}")


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "friction_factor": torch.stack([item["friction_factor"] for item in batch]),
        "material_factor": torch.stack([item["material_factor"] for item in batch]),
        "roughness_factor": torch.stack([item["roughness_factor"] for item in batch]),
        "class_label": [str(item["class_label"]) for item in batch],
        "image_path": [str(item["image_path"]) for item in batch],
    }
    mask_presence = ["valid_mask" in item for item in batch]
    if any(mask_presence) and not all(mask_presence):
        raise ValueError(
            "a batch cannot mix samples with and without explicit valid_mask"
        )
    if mask_presence and all(mask_presence):
        out["valid_mask"] = torch.stack(
            [item["valid_mask"] for item in batch]
        )
    secondary_keys = sorted(
        {
            key
            for item in batch
            for key in item
            if key.endswith("_image") and key != "image"
        }
    )
    for key in secondary_keys:
        presence = [key in item for item in batch]
        if not all(presence):
            raise ValueError(f"a batch cannot mix samples with and without {key}")
        out[key] = torch.stack([item[key] for item in batch])
    return out


class _EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch permutation with a resumable item offset."""

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        epoch: int,
        start_index: int = 0,
    ) -> None:
        self.size = max(int(size), 0)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = min(max(int(start_index), 0), self.size)

    def set_epoch(self, epoch: int) -> None:
        """Advance to a fresh deterministic order without rebuilding workers."""

        self.epoch = int(epoch)
        self.start_index = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1_000_003 * self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        return iter(order[self.start_index :])

    def __len__(self) -> int:
        return self.size - self.start_index


class _EpochWeightedSampler(Sampler[int]):
    """Deterministic weighted sampling whose full sequence can be sliced."""

    def __init__(
        self,
        weights: np.ndarray | torch.Tensor,
        *,
        num_samples: int,
        seed: int,
        epoch: int,
        start_index: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        self.num_samples = max(int(num_samples), 0)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = min(
            max(int(start_index), 0),
            self.num_samples,
        )
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ValueError("weighted sampler requires a non-empty 1D weight vector")
        if not bool(torch.isfinite(self.weights).all()) or not bool(
            self.weights.ge(0).all()
        ):
            raise ValueError("weighted sampler weights must be finite and non-negative")
        if not bool(self.weights.sum() > 0):
            raise ValueError("weighted sampler requires at least one positive weight")

    def set_epoch(self, epoch: int) -> None:
        """Advance to a fresh deterministic draw without rebuilding workers."""

        self.epoch = int(epoch)
        self.start_index = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1_000_003 * self.epoch)
        order = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(order[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index


def _set_train_loader_epoch(loader: DataLoader, epoch: int) -> bool:
    """Update an epoch-aware sampler in place.

    Returning ``False`` tells the caller that this loader uses a sampler whose
    epoch is baked into its construction (for example the factor-graph batch
    sampler), so rebuilding remains necessary.  The common natural and
    weighted samplers return ``True`` and therefore keep persistent Windows
    workers alive across epochs.
    """

    sampler = getattr(loader, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(int(epoch))
        return True
    batch_sampler = getattr(loader, "batch_sampler", None)
    set_batch_epoch = getattr(batch_sampler, "set_epoch", None)
    if callable(set_batch_epoch):
        set_batch_epoch(int(epoch))
        return True
    return False


def _release_dataloader_workers(loader: DataLoader) -> None:
    """Release persistent worker processes before memory-sensitive phases."""

    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    try:
        setattr(loader, "_iterator", None)
    except Exception:
        pass


class RSCDFactorGraphPairBatchSampler(Sampler[list[int]]):
    """Build batches that contain RSCD factor-graph neighbors and positives.

    Factor-graph metric learning needs same-batch structure: exact-class
    positives for the coupling token and one-axis-different neighbors for the
    friction/material/roughness graph. Plain class-balanced sampling is fair at
    epoch level but often misses the wet/water concrete hard subgraph inside a
    small batch. This sampler is therefore a task-adapted batch constructor, not
    a generic weak-class oversampler.
    """

    def __init__(
        self,
        dataset: RSCDSurfaceDataset,
        *,
        class_to_idx: dict[str, int],
        batch_size: int,
        num_samples: int,
        seed: int,
        pair_slots: int = 2,
        positive_slots: int = 1,
        wet_concrete_focus_scale: float = 3.0,
        roughness_focus_scale: float = 1.5,
        wet_water_focus_scale: float = 1.5,
        focus_pairs: list[str] | tuple[str, ...] | None = None,
        focus_pairs_only: bool = False,
        start_batch: int = 0,
    ) -> None:
        if batch_size < 2:
            raise ValueError("factor_graph_pair_sampling requires batch_size >= 2")
        self.dataset = dataset
        self.class_to_idx = {canonical_class_label(k): int(v) for k, v in class_to_idx.items()}
        self.idx_to_class = {int(v): canonical_class_label(k) for k, v in self.class_to_idx.items()}
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.num_batches = max(int(np.ceil(max(self.num_samples, 1) / float(self.batch_size))), 1)
        self.seed = int(seed)
        self.pair_slots = max(int(pair_slots), 0)
        self.positive_slots = max(int(positive_slots), 0)
        self.start_batch = min(max(int(start_batch), 0), self.num_batches)

        self.class_to_rows: dict[int, np.ndarray] = {}
        for label_name, group in dataset.df.groupby("class_label_canonical", sort=True):
            idx = self.class_to_idx.get(canonical_class_label(label_name))
            if idx is None:
                continue
            rows = group.index.to_numpy(dtype=np.int64)
            if rows.size > 0:
                self.class_to_rows[int(idx)] = rows
        self.present_classes = np.array(sorted(self.class_to_rows), dtype=np.int64)
        if self.present_classes.size == 0:
            raise ValueError("factor_graph_pair_sampling found no present classes in dataset")

        spec = build_rscd_factor_spec(self.class_to_idx)
        factors = spec.class_to_factor.numpy()
        wet_idx = FACTOR_LABELS["friction"].index("wet")
        water_idx = FACTOR_LABELS["friction"].index("water")
        concrete_idx = FACTOR_LABELS["material"].index("concrete")
        requested_pairs: set[frozenset[str]] = set()
        for item in focus_pairs or []:
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) == 2:
                requested_pairs.add(
                    frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))
                )

        pairs: list[tuple[int, int]] = []
        weights: list[float] = []
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            if left not in self.class_to_rows or right not in self.class_to_rows:
                continue
            pair_names = frozenset((self.idx_to_class[left], self.idx_to_class[right]))
            if requested_pairs and bool(focus_pairs_only) and pair_names not in requested_pairs:
                continue
            w = 1.0
            if requested_pairs and pair_names in requested_pairs:
                w *= 8.0
            left_f, left_m, _ = factors[left].tolist()
            right_f, right_m, _ = factors[right].tolist()
            both_wet_concrete = (
                left_f in {wet_idx, water_idx}
                and right_f in {wet_idx, water_idx}
                and left_m == concrete_idx
                and right_m == concrete_idx
            )
            if both_wet_concrete:
                w *= max(float(wet_concrete_focus_scale), 1.0)
            if pair.axis == "roughness":
                w *= max(float(roughness_focus_scale), 1.0)
            if pair.boundary == "wet_water":
                w *= max(float(wet_water_focus_scale), 1.0)
            pairs.append((left, right))
            weights.append(float(w))
        if not pairs:
            raise ValueError("factor_graph_pair_sampling found no valid factor graph pairs")
        self.pairs = np.asarray(pairs, dtype=np.int64)
        pair_weights = np.asarray(weights, dtype=np.float64)
        self.pair_probs = pair_weights / pair_weights.sum()

    def __len__(self) -> int:
        return int(max(self.num_batches - self.start_batch, 0))

    def _sample_row(self, cls_idx: int, rng: np.random.Generator) -> int:
        rows = self.class_to_rows[int(cls_idx)]
        return int(rows[int(rng.integers(0, len(rows)))])

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        for batch_i in range(self.num_batches):
            batch: list[int] = []
            max_pair_slots = min(self.pair_slots, self.batch_size // 2)
            for _pair_slot in range(max_pair_slots):
                pair_idx = int(rng.choice(len(self.pairs), p=self.pair_probs))
                left, right = self.pairs[pair_idx].tolist()
                batch.append(self._sample_row(left, rng))
                batch.append(self._sample_row(right, rng))
            remaining_after_pairs = self.batch_size - len(batch)
            max_positive_slots = min(self.positive_slots, remaining_after_pairs // 2)
            for _positive_slot in range(max_positive_slots):
                cls_idx = int(rng.choice(self.present_classes))
                batch.append(self._sample_row(cls_idx, rng))
                batch.append(self._sample_row(cls_idx, rng))
            while len(batch) < self.batch_size:
                cls_idx = int(rng.choice(self.present_classes))
                batch.append(self._sample_row(cls_idx, rng))
            rng.shuffle(batch)
            if batch_i < self.start_batch:
                continue
            yield batch


def _apply_anchor_error_sampler_weights(
    *,
    weights: np.ndarray,
    train_ds: RSCDSurfaceDataset,
    train_cfg: dict[str, Any],
    class_to_idx: dict[str, int],
) -> np.ndarray:
    """Boost cached anchor-error boundary samples for RSCD no-harm repair.

    This is a task-adapted JTT/GroupDRO sampler: it does not simply oversample
    a class. It oversamples samples in specific RSCD composite classes where a
    frozen anchor already makes mistakes, so PCGrad/no-harm losses see enough
    repair cases in each random batch.
    """

    cache_path_text = train_cfg.get("anchor_error_sampler_cache")
    if not cache_path_text:
        return weights
    cache_path = Path(str(cache_path_text))
    if not cache_path.exists():
        raise FileNotFoundError(f"anchor error sampler cache does not exist: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    image_paths = payload.get("image_paths")
    labels_obj = payload.get("labels")
    logits_obj = payload.get("logits")
    cache_class_to_idx = payload.get("class_to_idx", class_to_idx)
    if image_paths is None or labels_obj is None or logits_obj is None:
        raise ValueError(f"anchor error sampler cache must contain image_paths, labels, logits: {cache_path}")
    labels = torch.as_tensor(labels_obj, dtype=torch.long)
    logits = torch.as_tensor(logits_obj).float()
    if len(image_paths) != int(labels.numel()) or len(image_paths) != int(logits.shape[0]):
        raise ValueError(
            f"anchor error sampler cache length mismatch: paths={len(image_paths)} "
            f"labels={int(labels.numel())} logits={tuple(logits.shape)}"
        )

    cache_idx_to_class = {int(idx): canonical_class_label(name) for name, idx in dict(cache_class_to_idx).items()}
    focus_names = {
        canonical_class_label(name)
        for name in train_cfg.get("anchor_error_sampler_focus_classes", [])
    }
    if not focus_names:
        focus_names = {
            "water_concrete_slight",
            "water_concrete_severe",
            "water_concrete_smooth",
        }
    error_boost = max(float(train_cfg.get("anchor_error_sampler_error_boost", 1.0)), 1.0)
    focus_boost = max(float(train_cfg.get("anchor_error_sampler_focus_boost", 1.0)), 1.0)
    include_correct_focus = bool(train_cfg.get("anchor_error_sampler_include_correct_focus", True))

    boosts: dict[str, float] = {}
    stats = {
        "focus_cached": 0,
        "focus_errors": 0,
        "focus_correct": 0,
        "matched_dataset_rows": 0,
    }
    pred = logits.argmax(dim=1)
    for i, path_text in enumerate(image_paths):
        label_idx = int(labels[i])
        true_name = cache_idx_to_class.get(label_idx)
        if true_name is None or true_name not in focus_names:
            continue
        stats["focus_cached"] += 1
        is_error = int(pred[i]) != label_idx
        if is_error:
            stats["focus_errors"] += 1
            boost = error_boost
        else:
            stats["focus_correct"] += 1
            if not include_correct_focus:
                continue
            boost = focus_boost
        key = teacher_cache_key(str(path_text))
        boosts[key] = max(boosts.get(key, 1.0), float(boost))

    if not boosts:
        print(f"Anchor-error sampler cache loaded but no focus paths matched: {cache_path}")
        return weights

    boosted = weights.copy()
    image_path_series = train_ds.df["image_path"].astype(str)
    for row_idx, path_text in enumerate(image_path_series.tolist()):
        boost = boosts.get(teacher_cache_key(path_text))
        if boost is None:
            continue
        boosted[int(row_idx)] *= float(boost)
        stats["matched_dataset_rows"] += 1
    print(
        "Anchor-error sampler enabled: "
        f"cache={cache_path} focus_cached={stats['focus_cached']} "
        f"errors={stats['focus_errors']} correct={stats['focus_correct']} "
        f"matched_rows={stats['matched_dataset_rows']} "
        f"error_boost={error_boost:.1f} focus_boost={focus_boost:.1f}"
    )
    return boosted


def _verify_manifest_sha256(
    path: Path,
    expected_sha256: Any,
    *,
    split: str,
) -> str | None:
    """Verify a configured manifest digest before constructing its dataset.

    A missing digest keeps the legacy, opt-in behaviour.  Once a digest is
    configured, however, malformed values, missing files, and byte-level
    mismatches are all fatal.  This is intentionally separate from dataset
    parsing so a mismatched protocol cannot be partially consumed first.
    """

    if expected_sha256 is None:
        return None
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            f"data.{split}_manifest_sha256 must be exactly 64 hexadecimal "
            "characters"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"{split} manifest required by the frozen protocol does not exist: "
            f"{path}"
        )
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{split} manifest SHA256 mismatch before dataset construction: "
            f"path={path} expected={expected} actual={actual}"
        )
    print(
        f"Manifest SHA256 verified: split={split} path={path} sha256={actual}"
    )
    return actual


def _verify_train_dataset_contract(
    train_ds: RSCDSurfaceDataset,
    *,
    class_to_idx: dict[str, int],
    expected_samples: Any,
    expected_samples_per_class: Any,
) -> None:
    """Fail closed when the effective training dataset drifts from protocol."""

    if expected_samples is not None:
        expected_total = int(expected_samples)
        if expected_total < 0:
            raise ValueError("data.expected_train_samples must be non-negative")
        if len(train_ds) != expected_total:
            raise RuntimeError(
                "training protocol size mismatch: "
                f"expected={expected_total} actual={len(train_ds)}"
            )

    if expected_samples_per_class is None:
        return
    expected_per_class = int(expected_samples_per_class)
    if expected_per_class < 0:
        raise ValueError(
            "data.expected_train_samples_per_class must be non-negative"
        )
    counts = train_ds.df["class_label_canonical"].value_counts()
    mismatches: dict[str, dict[str, int]] = {}
    for class_name in sorted(class_to_idx):
        canonical_name = canonical_class_label(class_name)
        actual = int(counts.get(canonical_name, 0))
        if actual != expected_per_class:
            mismatches[canonical_name] = {
                "expected": expected_per_class,
                "actual": actual,
            }
    if mismatches:
        raise RuntimeError(
            "training protocol per-class size mismatch: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _resolve_data_input_geometry(
    data: Mapping[str, Any],
) -> tuple[int | tuple[int, int], tuple[int, int]]:
    """Resolve a legacy scalar or an explicit rectangular ``(H, W)`` view.

    ``data.image_size`` remains the authoritative legacy field and therefore
    keeps the historical transform constructor byte-for-byte unchanged when
    no rectangular keys are present.  Rectangular experiments must opt in with
    both ``input_height`` and ``input_width``; accepting only one would make a
    silently inferred geometry impossible to audit.
    """

    legacy_size = int(data.get("image_size", 192))
    if legacy_size <= 0:
        raise ValueError("data.image_size must be positive")
    raw_height = data.get("input_height")
    raw_width = data.get("input_width")
    if raw_height is None and raw_width is None:
        return legacy_size, (legacy_size, legacy_size)
    if raw_height is None or raw_width is None:
        raise ValueError(
            "rectangular input requires both data.input_height and "
            "data.input_width"
        )
    height, width = int(raw_height), int(raw_width)
    if height <= 0 or width <= 0:
        raise ValueError("data.input_height and data.input_width must be positive")
    return (height, width), (height, width)


_SYNTHETIC_BORDER_RESIZE_MODES = {
    "letterbox",
    "pad",
    "aspect_pad",
}


def _resolve_transform_support_contract(
    data: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve and fail closed on an opt-in full-support input contract.

    This does not change legacy transform behavior.  New native-input
    experiments can set ``data.require_full_valid_support`` so an inherited
    augmentation block cannot silently restore letterbox/affine fill pixels.
    ``data.require_direct_resize`` is the stricter geometry-screen contract:
    the first operation must be a direct stretch/resize and may not be
    replaced by a random crop.
    """

    augmentation_config = train_cfg.get("augmentation_config", {})
    if augmentation_config is None:
        augmentation_config = {}
    if not isinstance(augmentation_config, Mapping):
        raise TypeError("train.augmentation_config must be a mapping")
    train_aug_cfg = dict(augmentation_config)
    declared_train_mode = str(
        data.get("train_resize_mode", "letterbox")
    ).lower()
    declared_eval_mode = str(
        data.get("eval_resize_mode", "letterbox")
    ).lower()
    override_mode = train_aug_cfg.get("resize_mode")
    if override_mode is not None:
        effective_train_mode = str(override_mode).lower()
    else:
        effective_train_mode = declared_train_mode
        train_aug_cfg["resize_mode"] = effective_train_mode

    require_full_support = bool(
        data.get("require_full_valid_support", False)
    )
    require_direct_resize = bool(data.get("require_direct_resize", False))
    require_native_size = bool(data.get("require_native_size", False))
    if require_direct_resize and not require_full_support:
        raise ValueError(
            "data.require_direct_resize requires "
            "data.require_full_valid_support=true"
        )
    if require_native_size and not require_full_support:
        raise ValueError(
            "data.require_native_size requires "
            "data.require_full_valid_support=true"
        )
    if require_direct_resize and require_native_size:
        raise ValueError(
            "data.require_direct_resize and data.require_native_size are "
            "mutually exclusive"
        )
    if require_full_support:
        if (
            override_mode is not None
            and effective_train_mode != declared_train_mode
        ):
            raise ValueError(
                "full-support input contract forbids "
                "train.augmentation_config.resize_mode from overriding "
                "data.train_resize_mode"
            )
        if effective_train_mode in _SYNTHETIC_BORDER_RESIZE_MODES:
            raise ValueError(
                "full-support input contract forbids a padding train resize "
                f"mode: {effective_train_mode!r}"
            )
        if declared_eval_mode in _SYNTHETIC_BORDER_RESIZE_MODES:
            raise ValueError(
                "full-support input contract forbids a padding eval resize "
                f"mode: {declared_eval_mode!r}"
            )
        if bool(train_cfg.get("augmentation", True)):
            affine_degrees = float(
                train_aug_cfg.get("affine_degrees", 0.0)
            )
            affine_translate = tuple(
                float(value)
                for value in train_aug_cfg.get(
                    "affine_translate", [0.0, 0.0]
                )
            )
            affine_scale = tuple(
                float(value)
                for value in train_aug_cfg.get("affine_scale", [1.0, 1.0])
            )
            if (
                affine_degrees != 0.0
                or any(value != 0.0 for value in affine_translate)
                or affine_scale != (1.0, 1.0)
            ):
                raise ValueError(
                    "full-support input contract forbids RandomAffine because "
                    "its fill pixels recreate invalid borders"
                )
    if require_direct_resize:
        if effective_train_mode != "stretch" or declared_eval_mode != "stretch":
            raise ValueError(
                "direct-resize input contract requires train/eval resize_mode "
                "to be 'stretch'"
            )
        if bool(train_aug_cfg.get("random_resized_crop", False)):
            raise ValueError(
                "direct-resize input contract forbids random_resized_crop"
            )
    if require_native_size:
        if effective_train_mode != "native" or declared_eval_mode != "native":
            raise ValueError(
                "native-size input contract requires train/eval resize_mode "
                "to be 'native'"
            )
        if bool(train_aug_cfg.get("random_resized_crop", False)):
            raise ValueError(
                "native-size input contract forbids random_resized_crop"
            )

    audit = {
        "declared_train_resize_mode": declared_train_mode,
        "effective_train_resize_mode": effective_train_mode,
        "effective_eval_resize_mode": declared_eval_mode,
        "augmentation_resize_override_present": override_mode is not None,
        "require_full_valid_support": require_full_support,
        "require_direct_resize": require_direct_resize,
        "require_native_size": require_native_size,
    }
    return train_aug_cfg, audit


def build_loaders(
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    *,
    include_test: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """构造冻结 RSCD 数据加载器，并验证样本数/清单哈希/预处理契约。

    续训时 DataLoader 长度、batch size 与数据集大小还会和步级 checkpoint
    再比较一次，防止换电脑后因为清单错误而从不同 step 继续。
    """
    data = cfg["data"]
    train_cfg = cfg["train"]
    train_manifest = _runtime_path(data["train_manifest"])
    val_manifest = _runtime_path(data["val_manifest"])
    _verify_manifest_sha256(
        train_manifest,
        data.get("train_manifest_sha256"),
        split="train",
    )
    _verify_manifest_sha256(
        val_manifest,
        data.get("val_manifest_sha256"),
        split="val",
    )
    test_manifest: Path | None = None
    if include_test:
        # Keep every access to the official test manifest behind this guard.
        # Development runs with include_test=False must not even resolve or
        # hash the configured path.
        test_manifest = _runtime_path(data["test_manifest"])
        _verify_manifest_sha256(
            test_manifest,
            data.get("test_manifest_sha256"),
            split="test",
        )
    transform_size, input_hw = _resolve_data_input_geometry(data)
    train_aug_cfg, support_contract_audit = (
        _resolve_transform_support_contract(data, train_cfg)
    )
    explicit_valid_mask = bool(data.get("explicit_valid_mask", False))
    transform_builder = (
        build_transforms_with_valid_mask
        if explicit_valid_mask
        else build_transforms
    )
    train_tf = transform_builder(
        transform_size,
        train=bool(train_cfg.get("augmentation", True)),
        aug_cfg=train_aug_cfg,
    )
    eval_tf = transform_builder(
        transform_size,
        train=False,
        aug_cfg={
            "resize_mode": str(data.get("eval_resize_mode", "letterbox")),
            "resize_scale": float(data.get("eval_resize_scale", 1.14)),
        },
    )
    secondary_cfg = data.get("secondary_image_view")
    secondary_key: str | None = None
    secondary_train_tf = None
    secondary_eval_tf = None
    if secondary_cfg is not None:
        if not isinstance(secondary_cfg, Mapping):
            raise TypeError("data.secondary_image_view must be a mapping")
        if explicit_valid_mask:
            raise ValueError(
                "data.secondary_image_view is not yet compatible with "
                "data.explicit_valid_mask"
            )
        secondary_key = str(secondary_cfg.get("key", "observer_image"))
        if secondary_key != "observer_image":
            raise ValueError(
                "the current dual-view model contract requires "
                "data.secondary_image_view.key=observer_image"
            )
        secondary_height = int(secondary_cfg.get("input_height", 360))
        secondary_width = int(secondary_cfg.get("input_width", 240))
        secondary_size = (secondary_height, secondary_width)
        secondary_train_aug_cfg = copy.deepcopy(train_aug_cfg)
        secondary_train_aug_cfg["resize_mode"] = str(
            secondary_cfg.get("train_resize_mode", "native")
        )
        secondary_train_tf = build_transforms(
            secondary_size,
            train=bool(train_cfg.get("augmentation", True)),
            aug_cfg=secondary_train_aug_cfg,
        )
        secondary_eval_tf = build_transforms(
            secondary_size,
            train=False,
            aug_cfg={
                "resize_mode": str(
                    secondary_cfg.get("eval_resize_mode", "native")
                )
            },
        )
    # This is diagnostic metadata only; the user-facing geometry remains in
    # input_height/input_width.  Keeping it on the dataset makes shape audits
    # available without reaching back into a mutable config object.
    input_geometry_audit = {
        "input_hw": [int(input_hw[0]), int(input_hw[1])],
        "legacy_image_size": int(data.get("image_size", 192)),
        "train_resize_mode": str(data.get("train_resize_mode", "letterbox")),
        "eval_resize_mode": str(data.get("eval_resize_mode", "letterbox")),
        **support_contract_audit,
    }
    temporal_rotation_sampling = bool(
        train_cfg.get("temporal_rotation_sampling", False)
    )
    train_ds = RSCDSurfaceDataset(
        train_manifest,
        class_to_idx=class_to_idx,
        transform=train_tf,
        # The rotation sampler must see the complete manifest.  When enabled,
        # legacy limits are applied to representatives below, not destructively
        # to the dataset frame before class/second grouping.
        max_samples=(
            None
            if temporal_rotation_sampling
            else train_cfg.get("max_train_samples")
        ),
        max_samples_per_class=(
            None
            if temporal_rotation_sampling
            else train_cfg.get("max_train_samples_per_class")
        ),
        seed=int(cfg.get("seed", 79)),
        # Index substitution would silently replace a scheduled temporal key
        # or a wrong-sized native image with its neighbouring row.  Both cases
        # would make the requested training protocol differ from the samples
        # that actually reached the model, so fail closed for either contract.
        recover_unreadable=not (
            temporal_rotation_sampling or support_contract_audit["require_native_size"]
        ),
        emit_valid_mask=explicit_valid_mask,
        secondary_transform=secondary_train_tf,
        secondary_key=secondary_key or "observer_image",
    )
    train_ds.input_geometry_audit = copy.deepcopy(input_geometry_audit)
    _verify_train_dataset_contract(
        train_ds,
        class_to_idx=class_to_idx,
        expected_samples=data.get("expected_train_samples"),
        expected_samples_per_class=data.get(
            "expected_train_samples_per_class"
        ),
    )
    val_ds = RSCDSurfaceDataset(
        val_manifest,
        class_to_idx=class_to_idx,
        transform=eval_tf,
        max_samples=cfg["eval"].get("max_val_samples"),
        max_samples_per_class=cfg["eval"].get("max_val_samples_per_class"),
        seed=int(cfg.get("seed", 79)) + 1,
        recover_unreadable=False,
        emit_valid_mask=explicit_valid_mask,
        secondary_transform=secondary_eval_tf,
        secondary_key=secondary_key or "observer_image",
    )
    val_ds.input_geometry_audit = copy.deepcopy(input_geometry_audit)
    test_ds = (
        RSCDSurfaceDataset(
            test_manifest,
            class_to_idx=class_to_idx,
            transform=eval_tf,
            max_samples=cfg["eval"].get("max_test_samples"),
            max_samples_per_class=cfg["eval"].get("max_test_samples_per_class"),
            seed=int(cfg.get("seed", 79)) + 2,
            recover_unreadable=False,
            emit_valid_mask=explicit_valid_mask,
            secondary_transform=secondary_eval_tf,
            secondary_key=secondary_key or "observer_image",
        )
        if include_test
        else None
    )
    if test_ds is not None:
        test_ds.input_geometry_audit = copy.deepcopy(input_geometry_audit)
    expected_val_samples = data.get("expected_val_samples")
    if expected_val_samples is not None and len(val_ds) != int(expected_val_samples):
        raise RuntimeError(
            "validation protocol size mismatch: "
            f"expected={int(expected_val_samples)} actual={len(val_ds)}"
        )
    expected_test_samples = data.get("expected_test_samples")
    if (
        test_ds is not None
        and expected_test_samples is not None
        and len(test_ds) != int(expected_test_samples)
    ):
        raise RuntimeError(
            "test protocol size mismatch: "
            f"expected={int(expected_test_samples)} actual={len(test_ds)}"
        )
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    eval_cfg = cfg["eval"]
    eval_num_workers = int(eval_cfg.get("num_workers", num_workers))
    current_epoch = int(train_cfg.get("_current_epoch", 1) or 1)
    base_seed = int(cfg.get("seed", 79))
    train_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
        "collate_fn": collate,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        train_loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", False))
    eval_loader_kwargs = {
        "num_workers": eval_num_workers,
        "pin_memory": bool(
            eval_cfg.get(
                "pin_memory",
                train_cfg.get("pin_memory", torch.cuda.is_available()),
            )
        ),
        "collate_fn": collate,
    }
    if eval_num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = int(
            eval_cfg.get("prefetch_factor", train_cfg.get("prefetch_factor", 2))
        )
        eval_loader_kwargs["persistent_workers"] = bool(
            eval_cfg.get(
                "persistent_workers",
                train_cfg.get("persistent_workers", False),
            )
        )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(base_seed + 10_000_019 * current_epoch)
    val_generator = torch.Generator(device="cpu")
    val_generator.manual_seed(base_seed + 20_000_033)
    test_generator = torch.Generator(device="cpu")
    test_generator.manual_seed(base_seed + 30_000_047)
    sampler: Sampler[int] | None = None
    batch_sampler = None
    balanced_sampling_active = _balanced_sampling_active(train_cfg, current_epoch)
    resume_start_batch = int(train_cfg.get("_resume_start_step", train_cfg.get("resume_start_step", 0)) or 0)
    resume_start_index = int(resume_start_batch) * int(batch_size)
    if temporal_rotation_sampling:
        if bool(train_cfg.get("factor_graph_pair_sampling", False)):
            raise ValueError(
                "train.temporal_rotation_sampling cannot be combined with "
                "factor_graph_pair_sampling because the latter can repeat a "
                "class/acquisition-second key"
            )
        dedicated_total = train_cfg.get("temporal_rotation_max_samples")
        legacy_total = train_cfg.get("max_train_samples")
        samples_per_epoch = train_cfg.get("samples_per_epoch")
        for name, value in (
            ("temporal_rotation_max_samples", dedicated_total),
            ("max_train_samples", legacy_total),
            ("samples_per_epoch", samples_per_epoch),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"train.{name} must be non-negative")
        configured_totals = [
            int(value)
            for value in (dedicated_total, legacy_total, samples_per_epoch)
            if value is not None and int(value) > 0
        ]
        if len(set(configured_totals)) > 1:
            raise ValueError(
                "conflicting temporal rotation total limits: "
                "temporal_rotation_max_samples, max_train_samples, and "
                "samples_per_epoch must agree when more than one is set"
            )
        total_limit = configured_totals[0] if configured_totals else None
        dedicated_per_class = train_cfg.get(
            "temporal_rotation_max_samples_per_class"
        )
        legacy_per_class = train_cfg.get("max_train_samples_per_class")
        for name, value in (
            (
                "temporal_rotation_max_samples_per_class",
                dedicated_per_class,
            ),
            ("max_train_samples_per_class", legacy_per_class),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"train.{name} must be non-negative")
        configured_per_class = [
            int(value)
            for value in (dedicated_per_class, legacy_per_class)
            if value is not None and int(value) > 0
        ]
        if len(set(configured_per_class)) > 1:
            raise ValueError(
                "conflicting temporal rotation per-class limits: "
                "temporal_rotation_max_samples_per_class and "
                "max_train_samples_per_class must agree"
            )
        per_class_limit = (
            configured_per_class[0] if configured_per_class else None
        )
        per_class_targets = train_cfg.get(
            "temporal_rotation_samples_per_class"
        )
        if per_class_targets is not None:
            if not isinstance(per_class_targets, Mapping):
                raise TypeError(
                    "train.temporal_rotation_samples_per_class must be a mapping"
                )
            if per_class_limit is not None:
                raise ValueError(
                    "temporal_rotation_samples_per_class cannot be combined "
                    "with a scalar per-class limit"
                )
            per_class_targets = {
                str(label): int(value)
                for label, value in per_class_targets.items()
            }
            if total_limit is not None and sum(per_class_targets.values()) != total_limit:
                raise ValueError(
                    "the sum of temporal_rotation_samples_per_class must equal "
                    "the configured total samples per epoch"
                )
        sampler = EpochClassSecondRotationSampler(
            train_ds.df,
            seed=int(train_cfg.get("temporal_rotation_seed", base_seed)),
            epoch=current_epoch,
            start_index=resume_start_index,
            missing_timestamp=str(
                train_cfg.get("temporal_rotation_missing_timestamp", "error")
            ),
            max_samples_per_class=per_class_limit,
            samples_per_class=per_class_targets,
            max_samples=total_limit,
        )
        if sampler.start_index > 0:
            print(
                "Temporal rotation sampler resume enabled: "
                f"start_batch={resume_start_batch} "
                f"consumed={sampler.start_index} "
                f"remaining_samples={len(sampler)}"
            )
        print(
            "Temporal class-second rotation sampler enabled: "
            f"epoch={current_epoch} groups={sampler.class_second_groups} "
            f"global_samples={sampler.global_samples_per_epoch} "
            f"local_samples={len(sampler) + sampler.start_index} "
            f"rank={sampler.rank}/{sampler.num_replicas} "
            f"missing_timestamp={sampler.missing_timestamp}"
        )
    elif bool(train_cfg.get("factor_graph_pair_sampling", False)):
        num_samples = int(train_cfg.get("samples_per_epoch", 0)) or len(train_ds)
        batch_sampler = RSCDFactorGraphPairBatchSampler(
            train_ds,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=(
                int(train_cfg.get("factor_graph_pair_sampling_seed", base_seed))
                + 1_000_003 * current_epoch
            ),
            pair_slots=int(train_cfg.get("factor_graph_pair_sampling_pair_slots", 2)),
            positive_slots=int(train_cfg.get("factor_graph_pair_sampling_positive_slots", 1)),
            wet_concrete_focus_scale=float(train_cfg.get("factor_graph_pair_sampling_wet_concrete_focus_scale", 3.0)),
            roughness_focus_scale=float(train_cfg.get("factor_graph_pair_sampling_roughness_focus_scale", 1.5)),
            wet_water_focus_scale=float(train_cfg.get("factor_graph_pair_sampling_wet_water_focus_scale", 1.5)),
            focus_pairs=train_cfg.get("factor_graph_pair_sampling_focus_pairs"),
            focus_pairs_only=bool(train_cfg.get("factor_graph_pair_sampling_focus_pairs_only", False)),
            start_batch=resume_start_batch,
        )
        print(
            "Factor-graph pair batch sampler enabled: "
            f"batches={len(batch_sampler)} batch_size={batch_size} "
            f"start_batch={batch_sampler.start_batch}/{batch_sampler.num_batches} "
            f"pair_slots={batch_sampler.pair_slots} positive_slots={batch_sampler.positive_slots}"
        )
    elif balanced_sampling_active:
        sizes = train_ds.df.groupby("class_label_canonical")["class_label_canonical"].transform("size").astype(float)
        balance_power = float(train_cfg.get("balanced_sampling_power", 1.0))
        if not 0.0 <= balance_power <= 1.0:
            raise ValueError("train.balanced_sampling_power must be in [0, 1]")
        weights = sizes.clip(lower=1.0).pow(-balance_power).to_numpy(dtype=np.float64).copy()
        weights = _apply_anchor_error_sampler_weights(
            weights=weights,
            train_ds=train_ds,
            train_cfg=train_cfg,
            class_to_idx=class_to_idx,
        )
        num_samples = int(train_cfg.get("samples_per_epoch", 0)) or len(train_ds)
        sampler = _EpochWeightedSampler(
            weights,
            num_samples=num_samples,
            seed=int(train_cfg.get("balanced_sampling_seed", base_seed)),
            epoch=current_epoch,
            start_index=resume_start_index,
        )
        if sampler.start_index > 0:
            print(
                "Balanced sampler resume enabled: "
                f"start_batch={resume_start_batch} consumed={sampler.start_index} "
                f"remaining_samples={len(sampler)}/{num_samples}"
            )
        print(
            "Balanced sampler enabled: "
            f"epoch={current_epoch} power={balance_power:.3f} samples={num_samples}"
        )
    else:
        sampler = _EpochRandomSampler(
            len(train_ds),
            seed=int(train_cfg.get("natural_sampling_seed", base_seed)),
            epoch=current_epoch,
            start_index=resume_start_index,
        )
        if sampler.start_index > 0:
            print(
                "Natural sampler resume enabled: "
                f"start_batch={resume_start_batch} consumed={sampler.start_index} "
                f"remaining_samples={len(sampler)}/{len(train_ds)}"
            )
    if batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            generator=train_generator,
            **train_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            generator=train_generator,
            **train_loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(eval_cfg.get("batch_size", batch_size)),
        shuffle=False,
        generator=val_generator,
        **eval_loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=int(eval_cfg.get("batch_size", batch_size)),
            shuffle=False,
            generator=test_generator,
            **eval_loader_kwargs,
        )
        if test_ds is not None
        else None
    )
    test_size = str(len(test_ds)) if test_ds is not None else "not_loaded"
    mask_note = " explicit_valid_mask=true" if explicit_valid_mask else ""
    print(
        f"Dataset sizes: train={len(train_ds)} val={len(val_ds)} "
        f"test={test_size}{mask_note}"
    )
    return train_loader, val_loader, test_loader


def build_model(cfg: dict[str, Any], class_to_idx: dict[str, int]) -> nn.Module:
    """按配置实例化模型；正式发布配置只允许走 DREL-QRFME 分支。

    ``classifier_type=drel_rt`` 且 ``backbone_kwargs.variant=rfme`` 对应本文最终
    算法。模型从头训练，因此 ``pretrained`` 必须为 false。
    """
    # ``build_model`` is also a public programmatic boundary: callers may pass
    # an in-memory config without going through ``load_config``.
    _validate_arcq_composition_consistency_config(cfg)
    m = cfg["model"]
    head_cfg = m.get("head", {}) if isinstance(m.get("head"), dict) else {}
    head_type = str(m.get("head_type", head_cfg.get("type", "coupled_tensor")))
    backbone_kwargs = m.get("backbone_kwargs", {})
    if backbone_kwargs is None:
        backbone_kwargs = {}
    if not isinstance(backbone_kwargs, dict):
        raise TypeError("model.backbone_kwargs must be a mapping")
    classifier_type = str(m.get("classifier_type", m.get("model_type", "c3"))).strip().lower()
    if classifier_type in {"drel_rt", "drel-rt", "drel_rt_rfme", "drel-rt-rfme"}:
        # 当前正式实验会在这里直接返回；下方其他分支仅为历史实验兼容。
        if bool(m.get("pretrained", False)):
            raise ValueError("DREL-RT formal arms require model.pretrained=false")
        options = dict(backbone_kwargs)
        options.setdefault("variant", str(m.get("variant", "rfme")))
        options.setdefault("head_init_seed", int(m.get("head_init_seed", 970027)))
        return DRELRTSurfaceClassifier(
            class_to_idx,
            pretrained=False,
            **options,
        )
    if classifier_type in {
        "s7_drel_disagreement_router",
        "s7-drel-disagreement-router",
    }:
        anchor_checkpoint = Path(str(m["anchor_checkpoint"])).expanduser().resolve()
        expert_checkpoint = Path(str(m["drel_checkpoint"])).expanduser().resolve()
        checkpoint_specs = (
            (
                "anchor",
                anchor_checkpoint,
                str(m["anchor_expected_sha256"]).strip().lower(),
            ),
            (
                "drel",
                expert_checkpoint,
                str(m["drel_expected_sha256"]).strip().lower(),
            ),
        )
        checkpoint_audit: dict[str, Any] = {
            "architecture_version": S7DRELDisagreementRouter.architecture_version,
        }
        payloads: dict[str, Mapping[str, Any]] = {}
        for role, path, expected_sha256 in checkpoint_specs:
            if not path.is_file():
                raise FileNotFoundError(f"{role} checkpoint does not exist: {path}")
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"{role} checkpoint SHA256 mismatch: expected="
                    f"{expected_sha256} actual={actual_sha256}"
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("model"), Mapping
            ):
                raise TypeError(f"{role} checkpoint is missing model state")
            if payload.get("class_to_idx") != class_to_idx:
                raise RuntimeError(f"{role} checkpoint class_to_idx mismatch")
            payloads[role] = payload
            checkpoint_audit[role] = {
                "path": str(path),
                "sha256": actual_sha256,
                "epoch": int(payload.get("epoch", -1)),
                "state_keys": len(payload["model"]),
            }

        anchor_cfg = copy.deepcopy(cfg)
        anchor_cfg["model"]["classifier_type"] = "c3"
        anchor_cfg["model"]["backbone"] = (
            "convnext_tiny_gate_calibrated_tensor_coupling_concrete_film_rough_stem"
        )
        anchor_cfg["model"]["backbone_kwargs"] = {}
        anchor_cfg["model"]["use_backbone_factor_aux_heads"] = False
        anchor = build_model(anchor_cfg, class_to_idx)
        incompatible = anchor.load_state_dict(payloads["anchor"]["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict S7 anchor load returned incompatible keys")

        expert = BackboneLinearSurfaceClassifier(
            class_to_idx,
            backbone_name="arcq_road_drel_best_validated",
            out_dim=320,
            pretrained=False,
            backbone_kwargs={
                "study_mode": "no_regional_decomposition",
                "semantic_channels": (48, 96, 192, 320),
                "semantic_depths": (1, 2, 5, 2),
                "ledger_channels": (16, 24, 32, 48),
                "response_kernel_sizes": (7, 5, 3, 3),
                "group_width": 8,
                "orientations": 4,
                "simplex_smoothing": 0.05,
                "drop_path_rate": 0.10,
                "semantic_expansion": 2,
                "evidence_write_bound": 0.25,
            },
            dropout=0.0,
            head_init_seed=970027,
            apply_output_norm=False,
        )
        incompatible = expert.load_state_dict(payloads["drel"]["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict DREL expert load returned incompatible keys")
        return S7DRELDisagreementRouter(
            anchor,
            expert,
            anchor_feature_dim=int(m.get("anchor_feature_dim", 928)),
            expert_feature_dim=int(m.get("expert_feature_dim", 320)),
            projection_dim=int(m.get("router_projection_dim", 64)),
            hidden_dim=int(m.get("router_hidden_dim", 160)),
            residual_scale=float(m.get("router_residual_scale", 1.25)),
            rescue_gate_bias_init=float(
                m.get("router_rescue_gate_bias_init", -3.7)
            ),
            checkpoint_audit=checkpoint_audit,
        )
    if classifier_type in {"s7_minimal_core", "s7-minimal-core"}:
        return S7MinimalCoreSurfaceClassifier(
            class_to_idx,
            out_dim=int(m.get("embedding_dim", 768)),
            physics_dim=int(m.get("physics_dim", 96)),
            dropout=float(m.get("dropout", 0.2)),
        )
    if classifier_type in {"s7_research_core", "s7-research-core"}:
        # Unlike the compatibility-only Clean/Compact builders below, the
        # Research Core is constructed directly: its resolved graph and config
        # never instantiate or carry category-specific S7 correction metadata.
        return S7ResearchCoreSurfaceClassifier(
            class_to_idx,
            out_dim=int(m.get("embedding_dim", 768)),
            physics_dim=int(m.get("physics_dim", 96)),
            semantic_dim=int(m.get("semantic_physics_attention_dim", 64)),
            dropout=float(m.get("dropout", 0.2)),
        )
    if classifier_type in {"s7_clean", "s7-clean"}:
        # Construct the exact released architecture first so all non-tensor
        # route/spec metadata is derived from the resolved config, then retain
        # only the audited deployed-logit graph.  The returned module tree is
        # clean; the temporary legacy object is not retained.
        legacy_cfg = copy.deepcopy(cfg)
        legacy_cfg["model"]["classifier_type"] = "c3"
        legacy = build_model(legacy_cfg, class_to_idx)
        if not isinstance(legacy, C3FaRNetSurfaceClassifier):
            raise TypeError("s7_clean internal construction did not produce C3")
        return S7CleanSurfaceClassifier.from_legacy(legacy)
    if classifier_type in {
        "s7_clean_compact",
        "s7-clean-compact",
        "s7_lcami_product",
        "s7-lcami-product",
        "s7_lcami_additive",
        "s7-lcami-additive",
    }:
        # Build from the same SHA-locked S7 graph metadata, then physically
        # retain only the audited compact ancestry.  LCAMI arms differ solely
        # in the preregistered cross-window operator.
        legacy_cfg = copy.deepcopy(cfg)
        legacy_cfg["model"]["classifier_type"] = "c3"
        # These values exist only while reconstructing the locked parent graph
        # for deterministic state migration.  The returned compact module
        # physically deletes this validation-negative residual path, so clean
        # configs may truthfully declare it disabled.
        legacy_cfg["model"]["use_local_physics_field_branch"] = True
        legacy_cfg["model"]["local_physics_field_dim"] = 64
        legacy_cfg["model"]["local_physics_field_scale"] = 0.08
        legacy = build_model(legacy_cfg, class_to_idx)
        if not isinstance(legacy, C3FaRNetSurfaceClassifier):
            raise TypeError("S7 compact internal construction did not produce C3")
        if classifier_type in {"s7_clean_compact", "s7-clean-compact"}:
            return S7CleanCompactSurfaceClassifier.from_legacy(legacy)
        lcami_cfg = m.get("lcami", {})
        if lcami_cfg is None:
            lcami_cfg = {}
        if not isinstance(lcami_cfg, dict):
            raise TypeError("model.lcami must be a mapping")
        cross_mode = (
            "product"
            if classifier_type in {"s7_lcami_product", "s7-lcami-product"}
            else "additive"
        )
        return S7LCAMISurfaceClassifier.from_legacy(
            legacy,
            cross_mode=cross_mode,
            rank=int(lcami_cfg.get("rank", 24)),
            windows=tuple(
                int(value) for value in lcami_cfg.get("windows", (5, 9))
            ),
            residual_scale=float(lcami_cfg.get("residual_scale", 0.005)),
        )
    if classifier_type in {"arcq_l", "arcq_road_l"}:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "ARCQ-Road-L classifier does not provide pretrained weights; "
                "set pretrained=false"
            )
        return ARCQRoadLargeSurfaceClassifier(
            class_to_idx=class_to_idx,
            backbone_kwargs=dict(backbone_kwargs),
            out_dim=int(m.get("embedding_dim", 960)),
            dropout=float(m.get("dropout", 0.1)),
            use_aux_heads=bool(
                m.get("use_aux_heads", m.get("use_arcq_aux_heads", False))
            ),
        )
    if classifier_type == "arcq":
        if bool(m.get("pretrained", False)):
            raise ValueError("ARCQ classifier does not provide pretrained weights; set pretrained=false")
        return ARCQRoadSurfaceClassifier(
            class_to_idx=class_to_idx,
            backbone_kwargs=dict(backbone_kwargs),
            out_dim=int(m.get("embedding_dim", 768)),
            dropout=float(m.get("dropout", 0.1)),
            use_aux_heads=bool(m.get("use_aux_heads", m.get("use_arcq_aux_heads", False))),
            factor_head_mode=str(m.get("factor_head_mode", "legacy_coral")),
            factor_evidence_product_scale=float(
                m.get("factor_evidence_product_scale", 0.0)
            ),
            nullspace_graph_readout_mode=str(
                m.get("nullspace_graph_readout_mode", "disabled")
            ),
            classifier_fp32_logits=bool(
                m.get("classifier_fp32_logits", False)
            ),
        )
    if classifier_type in {"arcq_lite", "arcq_road_lite"}:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "ARCQ-Lite classifier does not provide pretrained weights; "
                "set pretrained=false"
            )
        return ARCQRoadLiteSurfaceClassifier(
            class_to_idx=class_to_idx,
            backbone_kwargs=dict(backbone_kwargs),
            out_dim=int(m.get("embedding_dim", 576)),
            dropout=float(m.get("dropout", 0.1)),
            use_aux_heads=bool(
                m.get("use_aux_heads", m.get("use_arcq_aux_heads", False))
            ),
        )
    if classifier_type in {"dualcnn", "same_parameter_dualcnn"}:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "Same-Parameter DualCNN does not provide pretrained weights"
            )
        return SameParameterDualCNNSurfaceClassifier(
            class_to_idx=class_to_idx,
            backbone_kwargs=dict(backbone_kwargs),
            out_dim=int(m.get("embedding_dim", 768)),
            dropout=float(m.get("dropout", 0.1)),
        )
    if classifier_type in {"principle_control", "arcq_principle_control"}:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "ARCQ principle controls do not provide pretrained weights"
            )
        control_type = m.get("control_type")
        backbone_name = str(m.get("backbone", "")).strip().lower()
        inferred_control = (
            principle_control_from_backbone_name(backbone_name)
            if backbone_name
            else None
        )
        if control_type is None:
            if inferred_control is None:
                raise ValueError(
                    "principle-control classifier requires model.control_type "
                    "or an arcq_control_* backbone name"
                )
            control_type = inferred_control
        elif inferred_control is not None and str(control_type).strip().lower() != inferred_control:
            raise ValueError(
                "model.control_type conflicts with model.backbone: "
                f"{control_type!r} versus {backbone_name!r}"
            )
        return PrincipleControlSurfaceClassifier(
            class_to_idx=class_to_idx,
            control_type=str(control_type),
            backbone_kwargs=dict(backbone_kwargs),
            out_dim=int(m.get("embedding_dim", 768)),
            dropout=float(m.get("dropout", 0.1)),
        )
    if classifier_type in {"backbone_linear", "linear_backbone"}:
        backbone_name = str(m.get("backbone", "")).strip()
        if not backbone_name:
            raise ValueError("backbone-linear classifier requires model.backbone")
        return BackboneLinearSurfaceClassifier(
            class_to_idx=class_to_idx,
            backbone_name=backbone_name,
            out_dim=int(m.get("embedding_dim", 768)),
            pretrained=bool(m.get("pretrained", False)),
            backbone_kwargs=dict(backbone_kwargs),
            dropout=float(m.get("dropout", 0.1)),
            head_init_seed=(
                int(m["head_init_seed"])
                if m.get("head_init_seed") is not None
                else None
            ),
            zero_head_bias=bool(m.get("zero_head_bias", False)),
            frozen_backbone_eval=bool(m.get("frozen_backbone_eval", False)),
            apply_output_norm=bool(m.get("apply_output_norm", True)),
            initialize_head_from_backbone_factors=bool(
                m.get("initialize_head_from_backbone_factors", False)
            ),
            use_roughness_coral_aux=bool(
                m.get("use_roughness_coral_aux", False)
            ),
            roughness_aux_scale=float(m.get("roughness_aux_scale", 1.0)),
        )
    if classifier_type in {
        "official_rspnet_external",
        "rspnet_official_external",
    }:
        return OfficialRSPNetSurfaceClassifier(
            class_to_idx=class_to_idx,
            pretrained=bool(m.get("pretrained", False)),
            **dict(backbone_kwargs),
        )
    if classifier_type in {"ceta_rspnet", "ceta_rspnet_released"}:
        if int(m.get("embedding_dim", 384)) != 384:
            raise ValueError(
                "released-head CETA-RSPNet requires model.embedding_dim=384"
            )
        return CETARSPNetSurfaceClassifier(
            class_to_idx=class_to_idx,
            backbone_kwargs=dict(backbone_kwargs),
        )
    if classifier_type in {
        "grit_rspnet_residual",
        "grit_rspnet_l_residual",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "GRIT-RSPNet residual loads its audited parent through "
                "model.backbone_kwargs.parent_checkpoint; set pretrained=false"
            )
        return GRITRSPNetResidualClassifier(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type in {
        "rsp_grit_ledger",
        "rspnet_grit_ledger",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "RSP-GRIT ledger loads its audited parent through "
                "model.backbone_kwargs.parent_checkpoint; set pretrained=false"
            )
        return RSPGRITLedgerClassifier(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type in {
        "rsp_logit_calibration_control",
        "rspnet_logit_calibration_control",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "RSP logit calibration control loads its audited parent "
                "through model.backbone_kwargs.parent_checkpoint; set "
                "pretrained=false"
            )
        return RSPLogitCalibrationControl(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type in {
        "rsp_gap_mlp_control",
        "rspnet_gap_mlp_control",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "RSP GAP MLP control loads its audited parent through "
                "model.backbone_kwargs.parent_checkpoint; set "
                "pretrained=false"
            )
        return RSPGAPMLPControl(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type in {
        "rsp_grit_spatial",
        "rspnet_grit_spatial",
        "rsp_grit_spatial_transport",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "RSP-GRIT spatial transport loads its audited parent "
                "through model.backbone_kwargs.parent_checkpoint; set "
                "pretrained=false"
            )
        if int(m.get("embedding_dim", 384)) != 384:
            raise ValueError(
                "released-head RSP-GRIT spatial transport requires "
                "model.embedding_dim=384"
            )
        return RSPGRITSpatialClassifier(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type in {
        "rsp_rcst_spatial",
        "rspnet_rcst_spatial",
        "rsp_rcst_spatial_transport",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "RSP-RCST spatial transport loads its audited parent "
                "through model.backbone_kwargs.parent_checkpoint; set "
                "pretrained=false"
            )
        if int(m.get("embedding_dim", 384)) != 384:
            raise ValueError(
                "released-head RSP-RCST spatial transport requires "
                "model.embedding_dim=384"
            )
        return RSPRCSTSpatialClassifier(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type in {
        "rsp_aort_spatial",
        "rspnet_aort_spatial",
        "rsp_aort_spatial_transport",
    }:
        if bool(m.get("pretrained", False)):
            raise ValueError(
                "RSP-AORT spatial transport loads its audited parent "
                "through model.backbone_kwargs.parent_checkpoint; set "
                "pretrained=false"
            )
        if int(m.get("embedding_dim", 384)) != 384:
            raise ValueError(
                "released-head RSP-AORT spatial transport requires "
                "model.embedding_dim=384"
            )
        return RSPAORTSpatialClassifier(
            class_to_idx=class_to_idx,
            **dict(backbone_kwargs),
        )
    if classifier_type not in {"c3", "c3_farnet"}:
        raise ValueError(f"unknown model.classifier_type: {classifier_type}")
    pareto_edge_expert_rules = m.get("pareto_edge_expert_rules")
    if m.get("pareto_edge_expert_rules_path"):
        pareto_edge_expert_rules = load_pareto_safe_logit_patch_rules(Path(str(m["pareto_edge_expert_rules_path"])))
    base_model = C3FaRNetSurfaceClassifier(
        class_to_idx=class_to_idx,
        backbone=str(m.get("backbone", "convnext_tiny")),
        embedding_dim=int(m.get("embedding_dim", 768)),
        pretrained=bool(m.get("pretrained", False)),
        backbone_kwargs=dict(backbone_kwargs),
        use_arcq_aux_heads=bool(m.get("use_arcq_aux_heads", False)),
        use_backbone_factor_aux_heads=bool(
            m.get("use_backbone_factor_aux_heads", False)
        ),
        dropout=float(m.get("dropout", 0.2)),
        token_dim=int(m.get("token_dim", 256)),
        pair_rank=int(m.get("pair_rank", 8)),
        triple_rank=int(m.get("triple_rank", 8)),
        head_type=head_type,
        hybrid_coupled_scale=float(m.get("hybrid_coupled_scale", 0.10)),
        gauge_fixed_coupling=bool(m.get("gauge_fixed_coupling", False)),
        coupling_gauge_epsilon=float(m.get("coupling_gauge_epsilon", 1.0e-6)),
        hardpair_correction_scale=float(m.get("hardpair_correction_scale", 0.08)),
        hardpair_margin_scale=float(m.get("hardpair_margin_scale", 0.18)),
        hardpair_gate_margin=float(m.get("hardpair_gate_margin", 1.00)),
        hardpair_gate_temperature=float(m.get("hardpair_gate_temperature", 4.00)),
        hardpair_error_gate_bias_init=float(m.get("hardpair_error_gate_bias_init", -3.5)),
        hardpair_error_gate_floor=float(m.get("hardpair_error_gate_floor", 0.0)),
        hardpair_physics_gate=str(m.get("hardpair_physics_gate", "none")),
        hardpair_physics_gate_floor=float(m.get("hardpair_physics_gate_floor", 0.0)),
        hardpair_physics_gate_power=float(m.get("hardpair_physics_gate_power", 1.0)),
        use_hardpair_value_signed_adapter=bool(m.get("use_hardpair_value_signed_adapter", False)),
        hardpair_value_adapter_pairs=m.get("hardpair_value_adapter_pairs"),
        hardpair_value_adapter_pair_scales=m.get("hardpair_value_adapter_pair_scales"),
        hardpair_value_adapter_hidden_dim=int(m.get("hardpair_value_adapter_hidden_dim", 48)),
        hardpair_value_adapter_scale=float(m.get("hardpair_value_adapter_scale", 0.10)),
        hardpair_value_adapter_gate_floor=float(m.get("hardpair_value_adapter_gate_floor", 0.0)),
        hardpair_value_adapter_value_aug_std=float(m.get("hardpair_value_adapter_value_aug_std", 0.0)),
        hardpair_value_adapter_dropout=float(m.get("hardpair_value_adapter_dropout", 0.0)),
        use_hardpair_value_rough_tail_guard=bool(m.get("use_hardpair_value_rough_tail_guard", False)),
        hardpair_value_rough_tail_guard_pairs=m.get("hardpair_value_rough_tail_guard_pairs"),
        hardpair_value_rough_tail_guard_threshold=float(m.get("hardpair_value_rough_tail_guard_threshold", 0.52)),
        hardpair_value_rough_tail_guard_temperature=float(m.get("hardpair_value_rough_tail_guard_temperature", 10.0)),
        hardpair_value_rough_tail_guard_strength=float(m.get("hardpair_value_rough_tail_guard_strength", 0.85)),
        hardpair_focus_classes=m.get("hardpair_focus_classes"),
        hardpair_focus_boundaries=m.get("hardpair_focus_boundaries"),
        hardpair_disabled_class_pairs=m.get("hardpair_disabled_class_pairs"),
        hardpair_pair_scales=m.get("hardpair_pair_scales"),
        hardpair_protected_classes=m.get("hardpair_protected_classes"),
        hardpair_sample_protect_classes=m.get("hardpair_sample_protect_classes"),
        hardpair_sample_protect_threshold=float(m.get("hardpair_sample_protect_threshold", 0.08)),
        hardpair_sample_protect_temperature=float(m.get("hardpair_sample_protect_temperature", 30.0)),
        boundary_use_physics_feature=bool(m.get("boundary_use_physics_feature", False)),
        use_boundary_experts=bool(m.get("use_boundary_experts", True)),
        use_physics_branch=bool(m.get("use_physics_branch", True)),
        physics_dim=int(m.get("physics_dim", 96)),
        physics_quality_cues=bool(m.get("physics_quality_cues", True)),
        physics_quality_region_cues=bool(m.get("physics_quality_region_cues", False)),
        use_semantic_physics_attention_branch=bool(m.get("use_semantic_physics_attention_branch", True)),
        semantic_physics_attention_dim=int(m.get("semantic_physics_attention_dim", 64)),
        use_local_physics_field_branch=bool(m.get("use_local_physics_field_branch", True)),
        local_physics_field_dim=int(m.get("local_physics_field_dim", 64)),
        local_physics_field_scale=float(m.get("local_physics_field_scale", 0.08)),
        use_physics_texture_stem_adapter=bool(m.get("use_physics_texture_stem_adapter", False)),
        physics_texture_stem_hidden_dim=int(m.get("physics_texture_stem_hidden_dim", 32)),
        physics_texture_stem_scale=float(m.get("physics_texture_stem_scale", 0.035)),
        physics_texture_stem_gate_floor=float(m.get("physics_texture_stem_gate_floor", 0.18)),
        use_scale_space_roughness_stem_adapter=bool(m.get("use_scale_space_roughness_stem_adapter", False)),
        scale_space_roughness_stem_hidden_dim=int(m.get("scale_space_roughness_stem_hidden_dim", 32)),
        scale_space_roughness_stem_scale=float(m.get("scale_space_roughness_stem_scale", 0.020)),
        scale_space_roughness_stem_gate_floor=float(m.get("scale_space_roughness_stem_gate_floor", 0.10)),
        scale_space_roughness_stem_gate_mode=str(m.get("scale_space_roughness_stem_gate_mode", "concrete_tail")),
        scale_space_roughness_stem_dry_tail_weight=float(m.get("scale_space_roughness_stem_dry_tail_weight", 1.0)),
        scale_space_roughness_stem_wet_hidden_tail_weight=float(
            m.get("scale_space_roughness_stem_wet_hidden_tail_weight", 1.0)
        ),
        use_pair_value_stem_conditioner=bool(m.get("use_pair_value_stem_conditioner", False)),
        pair_value_stem_hidden_dim=int(m.get("pair_value_stem_hidden_dim", 32)),
        pair_value_stem_scale=float(m.get("pair_value_stem_scale", 0.018)),
        pair_value_stem_gate_floor=float(m.get("pair_value_stem_gate_floor", 0.0)),
        pair_value_stem_value_aug_std=float(m.get("pair_value_stem_value_aug_std", 0.0)),
        pair_value_stem_learned_gate_bias=float(m.get("pair_value_stem_learned_gate_bias", -1.6)),
        use_wet_water_concrete_film_depth_stem_conditioner=bool(
            m.get("use_wet_water_concrete_film_depth_stem_conditioner", False)
        ),
        wet_water_concrete_film_depth_stem_hidden_dim=int(
            m.get("wet_water_concrete_film_depth_stem_hidden_dim", 36)
        ),
        wet_water_concrete_film_depth_stem_scale=float(
            m.get("wet_water_concrete_film_depth_stem_scale", 0.030)
        ),
        wet_water_concrete_film_depth_stem_gate_floor=float(
            m.get("wet_water_concrete_film_depth_stem_gate_floor", 0.04)
        ),
        wet_water_concrete_film_depth_stem_learned_gate_bias=float(
            m.get("wet_water_concrete_film_depth_stem_learned_gate_bias", -1.2)
        ),
        use_water_concrete_topology_texture_stem_conditioner=bool(
            m.get("use_water_concrete_topology_texture_stem_conditioner", False)
        ),
        water_concrete_topology_texture_stem_hidden_dim=int(
            m.get("water_concrete_topology_texture_stem_hidden_dim", 36)
        ),
        water_concrete_topology_texture_stem_scale=float(
            m.get("water_concrete_topology_texture_stem_scale", 0.026)
        ),
        water_concrete_topology_texture_stem_gate_floor=float(
            m.get("water_concrete_topology_texture_stem_gate_floor", 0.03)
        ),
        water_concrete_topology_texture_stem_learned_gate_bias=float(
            m.get("water_concrete_topology_texture_stem_learned_gate_bias", -1.25)
        ),
        use_scale_space_roughness_token_conditioner=bool(
            m.get("use_scale_space_roughness_token_conditioner", False)
        ),
        scale_space_roughness_token_hidden_dim=int(m.get("scale_space_roughness_token_hidden_dim", 64)),
        scale_space_roughness_token_scale=float(m.get("scale_space_roughness_token_scale", 0.10)),
        scale_space_roughness_token_gate_floor=float(m.get("scale_space_roughness_token_gate_floor", 0.0)),
        scale_space_roughness_token_dry_tail_weight=float(
            m.get("scale_space_roughness_token_dry_tail_weight", 1.0)
        ),
        scale_space_roughness_token_wet_hidden_tail_weight=float(
            m.get("scale_space_roughness_token_wet_hidden_tail_weight", 0.75)
        ),
        use_local_global_scale_token_conditioner=bool(
            m.get("use_local_global_scale_token_conditioner", False)
        ),
        local_global_scale_token_hidden_dim=int(m.get("local_global_scale_token_hidden_dim", 96)),
        local_global_scale_token_scale=float(m.get("local_global_scale_token_scale", 0.050)),
        local_global_scale_token_feature_scale=float(m.get("local_global_scale_token_feature_scale", 0.010)),
        local_global_scale_token_gate_floor=float(m.get("local_global_scale_token_gate_floor", 0.0)),
        local_global_scale_token_dropout=float(m.get("local_global_scale_token_dropout", 0.0)),
        local_global_scale_token_detach_context=bool(
            m.get("local_global_scale_token_detach_context", False)
        ),
        use_water_film_roughness_feature_film=bool(
            m.get("use_water_film_roughness_feature_film", False)
        ),
        water_film_roughness_feature_film_hidden_dim=int(
            m.get("water_film_roughness_feature_film_hidden_dim", 128)
        ),
        water_film_roughness_feature_film_scale=float(
            m.get("water_film_roughness_feature_film_scale", 0.080)
        ),
        water_film_roughness_feature_film_gate_floor=float(
            m.get("water_film_roughness_feature_film_gate_floor", 0.0)
        ),
        water_film_roughness_feature_film_max_gamma=float(
            m.get("water_film_roughness_feature_film_max_gamma", 0.18)
        ),
        water_film_roughness_feature_film_dropout=float(
            m.get("water_film_roughness_feature_film_dropout", 0.0)
        ),
        water_film_roughness_feature_film_detach_context=bool(
            m.get("water_film_roughness_feature_film_detach_context", False)
        ),
        use_pseudo_roughness_aware_reliability=bool(
            m.get("use_pseudo_roughness_aware_reliability", False)
        ),
        roughness_reliability_use_coupling_context=bool(
            m.get("roughness_reliability_use_coupling_context", False)
        ),
        pseudo_roughness_aware_reliability_hidden_dim=int(
            m.get("pseudo_roughness_aware_reliability_hidden_dim", 128)
        ),
        pseudo_roughness_aware_reliability_scale=float(
            m.get("pseudo_roughness_aware_reliability_scale", 0.060)
        ),
        pseudo_roughness_aware_reliability_rho_scale=float(
            m.get("pseudo_roughness_aware_reliability_rho_scale", 0.100)
        ),
        pseudo_roughness_aware_reliability_gate_floor=float(
            m.get("pseudo_roughness_aware_reliability_gate_floor", 0.0)
        ),
        pseudo_roughness_aware_reliability_dropout=float(
            m.get("pseudo_roughness_aware_reliability_dropout", 0.0)
        ),
        pseudo_roughness_aware_reliability_detach_context=bool(
            m.get("pseudo_roughness_aware_reliability_detach_context", False)
        ),
        use_spatial_factor_queries=bool(m.get("use_spatial_factor_queries", False)),
        spatial_factor_query_map_dim=int(m.get("spatial_factor_query_map_dim", 768)),
        spatial_factor_query_heads=int(m.get("spatial_factor_query_heads", 4)),
        spatial_factor_query_scale=float(m.get("spatial_factor_query_scale", 0.25)),
        use_dry_concrete_roughness_vor_residual=bool(m.get("use_dry_concrete_roughness_vor_residual", False)),
        dry_concrete_roughness_hidden_dim=int(m.get("dry_concrete_roughness_hidden_dim", 48)),
        dry_concrete_roughness_scale=float(m.get("dry_concrete_roughness_scale", 0.12)),
        dry_concrete_roughness_gate_threshold=float(m.get("dry_concrete_roughness_gate_threshold", 0.12)),
        dry_concrete_roughness_gate_temperature=float(m.get("dry_concrete_roughness_gate_temperature", 14.0)),
        use_dry_concrete_ordinal_chart_residual=bool(m.get("use_dry_concrete_ordinal_chart_residual", False)),
        dry_concrete_ordinal_chart_hidden_dim=int(m.get("dry_concrete_ordinal_chart_hidden_dim", 48)),
        dry_concrete_ordinal_chart_scale=float(m.get("dry_concrete_ordinal_chart_scale", 0.06)),
        dry_concrete_ordinal_chart_gate_threshold=float(m.get("dry_concrete_ordinal_chart_gate_threshold", 0.12)),
        dry_concrete_ordinal_chart_gate_temperature=float(m.get("dry_concrete_ordinal_chart_gate_temperature", 14.0)),
        dry_concrete_ordinal_chart_protect_confidence=float(
            m.get("dry_concrete_ordinal_chart_protect_confidence", 0.72)
        ),
        dry_concrete_ordinal_chart_protect_temperature=float(
            m.get("dry_concrete_ordinal_chart_protect_temperature", 18.0)
        ),
        use_dry_concrete_validation_transition=bool(m.get("use_dry_concrete_validation_transition", False)),
        dry_concrete_validation_transition_source=str(
            m.get("dry_concrete_validation_transition_source", "dry_concrete_severe")
        ),
        dry_concrete_validation_transition_target=str(
            m.get("dry_concrete_validation_transition_target", "dry_concrete_slight")
        ),
        dry_concrete_validation_transition_topk=int(m.get("dry_concrete_validation_transition_topk", 2)),
        dry_concrete_validation_transition_margin=float(m.get("dry_concrete_validation_transition_margin", 0.20)),
        dry_concrete_validation_transition_delta=float(m.get("dry_concrete_validation_transition_delta", 0.20)),
        use_backbone_isolated_dry_concrete_adapter=bool(
            m.get("use_backbone_isolated_dry_concrete_adapter", False)
        ),
        backbone_isolated_dry_concrete_branch_dim=int(
            m.get("backbone_isolated_dry_concrete_branch_dim", 96)
        ),
        backbone_isolated_dry_concrete_hidden_dim=int(
            m.get("backbone_isolated_dry_concrete_hidden_dim", 64)
        ),
        backbone_isolated_dry_concrete_scale=float(
            m.get("backbone_isolated_dry_concrete_scale", 0.18)
        ),
        backbone_isolated_dry_concrete_gate_threshold=float(
            m.get("backbone_isolated_dry_concrete_gate_threshold", 0.10)
        ),
        backbone_isolated_dry_concrete_gate_temperature=float(
            m.get("backbone_isolated_dry_concrete_gate_temperature", 14.0)
        ),
        backbone_isolated_dry_concrete_dropout=float(
            m.get("backbone_isolated_dry_concrete_dropout", 0.02)
        ),
        backbone_isolated_dry_concrete_output_mode=str(
            m.get("backbone_isolated_dry_concrete_output_mode", "free")
        ),
        use_dry_concrete_pair_signed_selector=bool(m.get("use_dry_concrete_pair_signed_selector", False)),
        dry_concrete_pair_selector_pairs=m.get("dry_concrete_pair_selector_pairs"),
        dry_concrete_pair_selector_hidden_dim=int(m.get("dry_concrete_pair_selector_hidden_dim", 48)),
        dry_concrete_pair_selector_shift_scale=float(m.get("dry_concrete_pair_selector_shift_scale", 0.65)),
        dry_concrete_pair_selector_gain_scale=float(m.get("dry_concrete_pair_selector_gain_scale", 0.50)),
        dry_concrete_pair_selector_direct_delta_scale=float(
            m.get("dry_concrete_pair_selector_direct_delta_scale", 0.0)
        ),
        dry_concrete_pair_selector_safe_margin=float(m.get("dry_concrete_pair_selector_safe_margin", 0.20)),
        dry_concrete_pair_selector_safe_temperature=float(
            m.get("dry_concrete_pair_selector_safe_temperature", 28.0)
        ),
        protected_factor_adapter_rank=int(m.get("protected_factor_adapter_rank", 6)),
        protected_factor_adapter_hidden_dim=int(m.get("protected_factor_adapter_hidden_dim", 96)),
        protected_factor_adapter_scale=float(m.get("protected_factor_adapter_scale", 0.08)),
        protected_factor_adapter_gate_margin=float(m.get("protected_factor_adapter_gate_margin", 0.18)),
        protected_factor_adapter_gate_temperature=float(m.get("protected_factor_adapter_gate_temperature", 10.0)),
        protected_factor_adapter_active_classes=m.get("protected_factor_adapter_active_classes"),
        protected_factor_adapter_protected_classes=m.get("protected_factor_adapter_protected_classes"),
        use_feature_value_boundary_corrector=bool(m.get("use_feature_value_boundary_corrector", False)),
        feature_value_boundary_pairs=m.get("feature_value_boundary_pairs"),
        feature_value_boundary_hidden_dim=int(m.get("feature_value_boundary_hidden_dim", 64)),
        feature_value_boundary_scale=float(m.get("feature_value_boundary_scale", 0.22)),
        feature_value_boundary_gate_margin=float(m.get("feature_value_boundary_gate_margin", 1.05)),
        feature_value_boundary_gate_temperature=float(m.get("feature_value_boundary_gate_temperature", 4.5)),
        feature_value_boundary_gate_floor=float(m.get("feature_value_boundary_gate_floor", 0.0)),
        feature_value_boundary_value_aug_std=float(m.get("feature_value_boundary_value_aug_std", 0.0)),
        feature_value_boundary_dropout=float(m.get("feature_value_boundary_dropout", 0.0)),
        feature_value_boundary_severe_tail_protect=bool(
            m.get("feature_value_boundary_severe_tail_protect", False)
        ),
        feature_value_boundary_severe_tail_protect_pairs=m.get(
            "feature_value_boundary_severe_tail_protect_pairs"
        ),
        feature_value_boundary_severe_tail_protect_strength=float(
            m.get("feature_value_boundary_severe_tail_protect_strength", 0.85)
        ),
        feature_value_boundary_severe_tail_protect_prob=float(
            m.get("feature_value_boundary_severe_tail_protect_prob", 0.34)
        ),
        feature_value_boundary_severe_tail_protect_tail_threshold=float(
            m.get("feature_value_boundary_severe_tail_protect_tail_threshold", 0.115)
        ),
        feature_value_boundary_severe_tail_protect_temperature=float(
            m.get("feature_value_boundary_severe_tail_protect_temperature", 16.0)
        ),
        use_water_concrete_opponent_feature_conditioner=bool(
            m.get("use_water_concrete_opponent_feature_conditioner", False)
        ),
        water_concrete_opponent_pairs=m.get("water_concrete_opponent_pairs"),
        water_concrete_opponent_hidden_dim=int(m.get("water_concrete_opponent_hidden_dim", 64)),
        water_concrete_opponent_scale=float(m.get("water_concrete_opponent_scale", 0.018)),
        water_concrete_opponent_gate_margin=float(m.get("water_concrete_opponent_gate_margin", 1.08)),
        water_concrete_opponent_gate_temperature=float(
            m.get("water_concrete_opponent_gate_temperature", 4.5)
        ),
        water_concrete_opponent_gate_floor=float(m.get("water_concrete_opponent_gate_floor", 0.03)),
        water_concrete_opponent_value_aug_std=float(m.get("water_concrete_opponent_value_aug_std", 0.0)),
        water_concrete_opponent_dropout=float(m.get("water_concrete_opponent_dropout", 0.0)),
        use_factor_graph_edge_flow_corrector=bool(m.get("use_factor_graph_edge_flow_corrector", False)),
        factor_graph_edge_flow_pairs=m.get("factor_graph_edge_flow_pairs"),
        factor_graph_edge_flow_hidden_dim=int(m.get("factor_graph_edge_flow_hidden_dim", 64)),
        factor_graph_edge_flow_scale=float(m.get("factor_graph_edge_flow_scale", 0.10)),
        factor_graph_edge_flow_gate_margin=float(m.get("factor_graph_edge_flow_gate_margin", 0.90)),
        factor_graph_edge_flow_gate_temperature=float(m.get("factor_graph_edge_flow_gate_temperature", 4.0)),
        factor_graph_edge_flow_gate_floor=float(m.get("factor_graph_edge_flow_gate_floor", 0.0)),
        factor_graph_edge_flow_confidence_protect=float(m.get("factor_graph_edge_flow_confidence_protect", 0.74)),
        factor_graph_edge_flow_confidence_temperature=float(
            m.get("factor_graph_edge_flow_confidence_temperature", 16.0)
        ),
        factor_graph_edge_flow_dropout=float(m.get("factor_graph_edge_flow_dropout", 0.0)),
        use_tristate_wet_concrete_boundary_expert=bool(
            m.get("use_tristate_wet_concrete_boundary_expert", False)
        ),
        tristate_wet_concrete_boundary_pairs=m.get("tristate_wet_concrete_boundary_pairs"),
        tristate_wet_concrete_boundary_hidden_dim=int(
            m.get("tristate_wet_concrete_boundary_hidden_dim", 64)
        ),
        tristate_wet_concrete_boundary_scale=float(
            m.get("tristate_wet_concrete_boundary_scale", 0.08)
        ),
        tristate_wet_concrete_boundary_gate_margin=float(
            m.get("tristate_wet_concrete_boundary_gate_margin", 0.85)
        ),
        tristate_wet_concrete_boundary_gate_temperature=float(
            m.get("tristate_wet_concrete_boundary_gate_temperature", 5.0)
        ),
        tristate_wet_concrete_boundary_gate_floor=float(
            m.get("tristate_wet_concrete_boundary_gate_floor", 0.0)
        ),
        tristate_wet_concrete_boundary_confidence_protect=float(
            m.get("tristate_wet_concrete_boundary_confidence_protect", 0.78)
        ),
        tristate_wet_concrete_boundary_confidence_temperature=float(
            m.get("tristate_wet_concrete_boundary_confidence_temperature", 16.0)
        ),
        tristate_wet_concrete_boundary_dropout=float(
            m.get("tristate_wet_concrete_boundary_dropout", 0.0)
        ),
        tristate_wet_concrete_boundary_severe_protect=bool(
            m.get("tristate_wet_concrete_boundary_severe_protect", False)
        ),
        tristate_wet_concrete_boundary_severe_protect_prob=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_prob", 0.30)
        ),
        tristate_wet_concrete_boundary_severe_protect_raw_margin=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_raw_margin", 0.0)
        ),
        tristate_wet_concrete_boundary_severe_protect_temperature=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_temperature", 12.0)
        ),
        tristate_wet_concrete_boundary_severe_protect_strength=float(
            m.get("tristate_wet_concrete_boundary_severe_protect_strength", 1.0)
        ),
        use_closed_set_factor_redistributor=bool(m.get("use_closed_set_factor_redistributor", False)),
        closed_set_factor_redistributor_sets=m.get("closed_set_factor_redistributor_sets"),
        closed_set_factor_redistributor_hidden_dim=int(m.get("closed_set_factor_redistributor_hidden_dim", 96)),
        closed_set_factor_redistributor_scale=float(m.get("closed_set_factor_redistributor_scale", 0.06)),
        closed_set_factor_redistributor_gate_floor=float(m.get("closed_set_factor_redistributor_gate_floor", 0.0)),
        closed_set_factor_redistributor_mass_threshold=float(
            m.get("closed_set_factor_redistributor_mass_threshold", 0.08)
        ),
        closed_set_factor_redistributor_margin_threshold=float(
            m.get("closed_set_factor_redistributor_margin_threshold", 0.25)
        ),
        closed_set_factor_redistributor_temperature=float(m.get("closed_set_factor_redistributor_temperature", 8.0)),
        closed_set_factor_redistributor_dropout=float(m.get("closed_set_factor_redistributor_dropout", 0.0)),
        closed_set_factor_redistributor_gate_bias_init=float(
            m.get("closed_set_factor_redistributor_gate_bias_init", -2.5)
        ),
        closed_set_factor_redistributor_use_graph_locality_guard=bool(
            m.get("closed_set_factor_redistributor_use_graph_locality_guard", False)
        ),
        closed_set_factor_redistributor_graph_max_distance=float(
            m.get("closed_set_factor_redistributor_graph_max_distance", 2.0)
        ),
        closed_set_factor_redistributor_graph_guard_floor=float(
            m.get("closed_set_factor_redistributor_graph_guard_floor", 0.0)
        ),
        closed_set_factor_redistributor_graph_guard_temperature=float(
            m.get("closed_set_factor_redistributor_graph_guard_temperature", 12.0)
        ),
        use_backbone_family_ordinal_no_spill_adapter=bool(
            m.get("use_backbone_family_ordinal_no_spill_adapter", False)
        ),
        backbone_family_ordinal_no_spill_hidden_dim=int(
            m.get("backbone_family_ordinal_no_spill_hidden_dim", 96)
        ),
        backbone_family_ordinal_no_spill_family_embed_dim=int(
            m.get("backbone_family_ordinal_no_spill_family_embed_dim", 12)
        ),
        backbone_family_ordinal_no_spill_scale=float(
            m.get("backbone_family_ordinal_no_spill_scale", 0.18)
        ),
        backbone_family_ordinal_no_spill_gate_threshold=float(
            m.get("backbone_family_ordinal_no_spill_gate_threshold", 0.055)
        ),
        backbone_family_ordinal_no_spill_gate_temperature=float(
            m.get("backbone_family_ordinal_no_spill_gate_temperature", 10.0)
        ),
        backbone_family_ordinal_no_spill_dropout=float(
            m.get("backbone_family_ordinal_no_spill_dropout", 0.02)
        ),
        backbone_family_ordinal_no_spill_families=m.get("backbone_family_ordinal_no_spill_families"),
        use_pair_value_mechanism_conditioner=bool(m.get("use_pair_value_mechanism_conditioner", False)),
        pair_value_mechanism_hidden_dim=int(m.get("pair_value_mechanism_hidden_dim", 64)),
        pair_value_mechanism_feature_scale=float(m.get("pair_value_mechanism_feature_scale", 0.010)),
        pair_value_mechanism_token_scale=float(m.get("pair_value_mechanism_token_scale", 0.060)),
        pair_value_mechanism_gate_floor=float(m.get("pair_value_mechanism_gate_floor", 0.0)),
        pair_value_mechanism_value_aug_std=float(m.get("pair_value_mechanism_value_aug_std", 0.0)),
        pair_value_mechanism_protect_classes=m.get("pair_value_mechanism_protect_classes"),
        pair_value_mechanism_protect_threshold=float(m.get("pair_value_mechanism_protect_threshold", 0.18)),
        pair_value_mechanism_protect_temperature=float(m.get("pair_value_mechanism_protect_temperature", 24.0)),
        use_coupled_form_expert_conditioner=bool(m.get("use_coupled_form_expert_conditioner", False)),
        coupled_form_expert_hidden_dim=int(m.get("coupled_form_expert_hidden_dim", 64)),
        coupled_form_expert_feature_scale=float(m.get("coupled_form_expert_feature_scale", 0.010)),
        coupled_form_expert_token_scale=float(m.get("coupled_form_expert_token_scale", 0.060)),
        coupled_form_expert_gate_floor=float(m.get("coupled_form_expert_gate_floor", 0.0)),
        coupled_form_expert_value_aug_std=float(m.get("coupled_form_expert_value_aug_std", 0.0)),
        coupled_form_expert_learned_gate_bias=float(m.get("coupled_form_expert_learned_gate_bias", -1.5)),
        coupled_form_expert_detach_context=bool(m.get("coupled_form_expert_detach_context", False)),
        coupled_form_expert_protect_classes=m.get("coupled_form_expert_protect_classes"),
        coupled_form_expert_protect_threshold=float(m.get("coupled_form_expert_protect_threshold", 0.18)),
        coupled_form_expert_protect_temperature=float(m.get("coupled_form_expert_protect_temperature", 24.0)),
        use_pareto_edge_expert=bool(m.get("use_pareto_edge_expert", False)),
        pareto_edge_expert_rules=pareto_edge_expert_rules,
        pareto_edge_expert_hidden_dim=int(m.get("pareto_edge_expert_hidden_dim", 48)),
        pareto_edge_expert_scale=float(m.get("pareto_edge_expert_scale", 1.0)),
        pareto_edge_expert_gate_temperature=float(m.get("pareto_edge_expert_gate_temperature", 18.0)),
        pareto_edge_expert_gate_floor=float(m.get("pareto_edge_expert_gate_floor", 0.0)),
        pareto_edge_expert_learned_gate_bias=float(m.get("pareto_edge_expert_learned_gate_bias", -1.6)),
        pareto_edge_expert_dropout=float(m.get("pareto_edge_expert_dropout", 0.0)),
        use_source_reliable_boundary_router=bool(m.get("use_source_reliable_boundary_router", False)),
        source_reliable_boundary_routes=m.get("source_reliable_boundary_routes"),
        source_reliable_boundary_hidden_dim=int(m.get("source_reliable_boundary_hidden_dim", 32)),
        source_reliable_boundary_scale=float(m.get("source_reliable_boundary_scale", 0.012)),
        source_reliable_boundary_gate_temperature=float(m.get("source_reliable_boundary_gate_temperature", 6.0)),
        source_reliable_boundary_physics_gate_floor=float(
            m.get("source_reliable_boundary_physics_gate_floor", 0.0)
        ),
        source_reliable_boundary_base_strength=float(m.get("source_reliable_boundary_base_strength", 0.0)),
        source_reliable_boundary_source_temperature=float(
            m.get("source_reliable_boundary_source_temperature", 28.0)
        ),
        source_reliable_boundary_learned_gate_bias=float(
            m.get("source_reliable_boundary_learned_gate_bias", -2.0)
        ),
        source_reliable_boundary_dropout=float(m.get("source_reliable_boundary_dropout", 0.0)),
        expose_hardpair_pair_value_evidence=bool(m.get("expose_hardpair_pair_value_evidence", False)),
    )
    use_cfor = bool(m.get("use_certified_factor_odds_reconciliation", False))
    use_rfbt = bool(m.get("use_reliable_factor_boundary_transport", False))
    if use_cfor and use_rfbt:
        raise ValueError("CFOR and RFBT are mutually exclusive inference wrappers")
    if use_cfor:
        raw_counts = m.get("certified_factor_odds_reconciliation_class_counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError(
                "model.certified_factor_odds_reconciliation_class_counts must "
                "be a class-name mapping when CFOR is enabled"
            )
        return CertifiedFactorOddsReconciliationClassifier(
            base_model,
            raw_counts,
        )
    if use_rfbt:
        raw_counts = m.get("reliable_factor_boundary_transport_class_counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError(
                "model.reliable_factor_boundary_transport_class_counts must "
                "be a class-name mapping when RFBT is enabled"
            )
        return ReliableFactorBoundaryTransportClassifier(
            base_model,
            raw_counts,
        )
    return base_model


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA256 digest without buffering the file in RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def flexible_load(
    model: nn.Module,
    checkpoint: str | None,
    *,
    skip_prefixes: list[str] | tuple[str, ...] | None = None,
    expected_sha256: str | None = None,
    immutable_prefixes: list[str] | tuple[str, ...] | None = None,
    expected_class_to_idx: dict[str, int] | None = None,
    require_class_to_idx: bool = False,
    required_loaded_keys: list[str] | tuple[str, ...] | None = None,
    allowed_missing_prefixes: list[str] | tuple[str, ...] | None = None,
    allowed_skipped_prefixes: list[str] | tuple[str, ...] | None = None,
    expected_loaded_key_count: int | None = None,
) -> dict[str, Any]:
    if not checkpoint:
        return {
            "loaded": 0,
            "skipped": 0,
            "missing": 0,
            "unexpected": 0,
            "path": None,
            "checkpoint_sha256": None,
            "expected_sha256": None,
            "loaded_keys": [],
            "skipped_keys": [],
            "missing_keys": [],
            "unexpected_keys": [],
            "immutable_verified_keys": [],
            "class_to_idx_verified": False,
            "arcq_versionless_v1_weights_only": False,
        }
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    expected_digest = str(expected_sha256 or "").strip().lower()
    if expected_digest and (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    actual_digest = _sha256_file(path)
    if expected_digest and actual_digest != expected_digest:
        raise RuntimeError(
            "checkpoint SHA256 mismatch before flexible load: "
            f"path={path} expected={expected_digest} actual={actual_digest}"
        )
    skip_prefixes = tuple(str(prefix) for prefix in (skip_prefixes or ()))
    immutable_prefixes = tuple(
        str(prefix) for prefix in (immutable_prefixes or ())
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_class_to_idx = state.get("class_to_idx") if isinstance(state, dict) else None
    if require_class_to_idx and not isinstance(checkpoint_class_to_idx, dict):
        raise RuntimeError(
            "checkpoint is missing the required class_to_idx mapping: "
            f"path={path}"
        )
    class_to_idx_verified = False
    if expected_class_to_idx is not None and checkpoint_class_to_idx is not None:
        normalized_checkpoint_map = {
            str(name): int(index) for name, index in checkpoint_class_to_idx.items()
        }
        normalized_expected_map = {
            str(name): int(index) for name, index in expected_class_to_idx.items()
        }
        if normalized_checkpoint_map != normalized_expected_map:
            raise RuntimeError(
                "checkpoint class_to_idx does not match the active dataset mapping: "
                f"path={path}"
            )
        class_to_idx_verified = True
    raw = state.get("model", state.get("state_dict", state))
    target = model.state_dict()
    target_has_arcq = any(
        name.endswith("._arcq_state_version") or name == "_arcq_state_version"
        for name in target
    )
    versionless_v1_terminal = target_has_arcq and any(
        ".h_stages.3." in name or name.startswith("h_stages.3.")
        for name in raw
    ) and not any(
        name.endswith("._arcq_state_version") or name == "_arcq_state_version"
        for name in raw
    )
    if versionless_v1_terminal:
        warnings.warn(
            "flexible_load detected a versionless ARCQ v1 payload. This path "
            "is weights-only and may skip the dead terminal-H keys; it is not "
            "an audited exact migration or optimizer resume. Use strict loading "
            "for a complete ARCQ migration audit.",
            UserWarning,
            stacklevel=2,
        )
    loadable = {}
    skipped = []
    immutable_verified: list[str] = []
    aliases = {
        "classifier.weight": "linear_head.weight",
        "classifier.bias": "linear_head.bias",
        "backbone.proj.weight": "backbone.global_proj.weight",
        "backbone.proj.bias": "backbone.global_proj.bias",
    }
    def state_entry_compatible(current: Any, incoming: Any) -> bool:
        current_is_tensor = isinstance(current, torch.Tensor)
        incoming_is_tensor = isinstance(incoming, torch.Tensor)
        if current_is_tensor and incoming_is_tensor:
            return tuple(current.shape) == tuple(incoming.shape)
        return not current_is_tensor and not incoming_is_tensor

    for name, tensor in raw.items():
        if immutable_prefixes and any(
            name.startswith(prefix) for prefix in immutable_prefixes
        ):
            if name not in target or not state_entry_compatible(
                target[name], tensor
            ):
                raise RuntimeError(
                    "immutable checkpoint key is missing from the initialized model or has "
                    f"a different state kind/shape: {name}"
                )
            if isinstance(target[name], torch.Tensor):
                current = target[name].detach().to(device="cpu")
                incoming = tensor.detach().to(device="cpu")
                equal = torch.equal(current, incoming)
                maximum_error = (
                    float(
                        (current.float() - incoming.float()).abs().max().item()
                    )
                    if not equal
                    else 0.0
                )
            else:
                equal = target[name] == tensor
                maximum_error = None
            if not equal:
                raise RuntimeError(
                    "checkpoint would overwrite an immutable initialized parameter: "
                    f"key={name} max_abs_error={maximum_error}"
                )
            immutable_verified.append(name)
            continue
        if skip_prefixes and any(name.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(name)
            continue
        if name.endswith(("cell_mask", "chart_mask", "active_mask")):
            skipped.append(name)
            continue
        load_name = name
        if load_name in target and state_entry_compatible(target[load_name], tensor):
            loadable[load_name] = tensor
            continue
        alias_name = aliases.get(name)
        if (
            alias_name
            and alias_name in target
            and state_entry_compatible(target[alias_name], tensor)
        ):
            loadable[alias_name] = tensor
        else:
            skipped.append(name)
    expected_immutable = {
        name
        for name in target
        if immutable_prefixes
        and any(name.startswith(prefix) for prefix in immutable_prefixes)
    }
    if set(immutable_verified) != expected_immutable:
        absent = sorted(expected_immutable - set(immutable_verified))
        raise RuntimeError(
            "checkpoint did not verify every immutable initialized parameter; "
            f"missing_keys={absent}"
        )
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    loaded_keys = sorted(loadable)
    skipped_keys = sorted(skipped)
    missing_keys = sorted(missing)
    unexpected_keys = sorted(unexpected)

    required_loaded = tuple(str(key) for key in (required_loaded_keys or ()))
    absent_required = sorted(set(required_loaded) - set(loaded_keys))
    if absent_required:
        raise RuntimeError(
            "flexible checkpoint failed the required-loaded-key contract: "
            f"missing={absent_required}"
        )
    if expected_loaded_key_count is not None and len(loaded_keys) != int(
        expected_loaded_key_count
    ):
        raise RuntimeError(
            "flexible checkpoint loaded an unexpected number of keys: "
            f"expected={int(expected_loaded_key_count)} actual={len(loaded_keys)}"
        )
    if allowed_missing_prefixes is not None:
        allowed_missing = tuple(str(prefix) for prefix in allowed_missing_prefixes)
        disallowed_missing = [
            name
            for name in missing_keys
            if not any(name.startswith(prefix) for prefix in allowed_missing)
        ]
        if disallowed_missing:
            raise RuntimeError(
                "flexible checkpoint has disallowed missing model keys: "
                f"{disallowed_missing}"
            )
    if allowed_skipped_prefixes is not None:
        allowed_skipped = tuple(str(prefix) for prefix in allowed_skipped_prefixes)
        disallowed_skipped = [
            name
            for name in skipped_keys
            if not any(name.startswith(prefix) for prefix in allowed_skipped)
        ]
        if disallowed_skipped:
            raise RuntimeError(
                "flexible checkpoint has disallowed skipped source keys: "
                f"{disallowed_skipped}"
            )
    if unexpected_keys:
        raise RuntimeError(
            "flexible checkpoint produced unexpected model keys: "
            f"{unexpected_keys}"
        )
    print(f"Loaded flexible checkpoint: {path}")
    if skip_prefixes:
        print(f"  skip_prefixes={list(skip_prefixes)}")
    print(f"  loaded={len(loadable)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
    return {
        "loaded": len(loadable),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "path": str(path),
        "checkpoint_sha256": actual_digest,
        "expected_sha256": expected_digest or None,
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "immutable_verified_keys": sorted(immutable_verified),
        "class_to_idx_verified": bool(class_to_idx_verified),
        "arcq_versionless_v1_weights_only": bool(versionless_v1_terminal),
    }


def strict_model_state_load(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    expected_class_to_idx: dict[str, int],
    expected_state_key_count: int | None = None,
    expected_classifier_type: str | None = None,
) -> dict[str, Any]:
    """Load a same-schema model checkpoint without migration or skipped state.

    This is intentionally separate from ``flexible_load``.  It is the
    fail-closed path for experiments whose source checkpoint was exported for
    the exact target class: every state entry, including deterministic masks,
    must have the same name, kind, shape and dtype and must compare bitwise
    equal after loading.
    """

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_digest = str(expected_sha256).strip().lower()
    if (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    actual_digest = _sha256_file(path)
    if actual_digest != expected_digest:
        raise RuntimeError(
            "checkpoint SHA256 mismatch before strict model-state load: "
            f"path={path} expected={expected_digest} actual={actual_digest}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("strict model-state checkpoint must contain a model mapping")
    checkpoint_class_to_idx = payload.get("class_to_idx")
    if not isinstance(checkpoint_class_to_idx, dict):
        raise RuntimeError("strict model-state checkpoint is missing class_to_idx")
    normalized_source_classes = {
        str(name): int(index) for name, index in checkpoint_class_to_idx.items()
    }
    normalized_target_classes = {
        str(name): int(index) for name, index in expected_class_to_idx.items()
    }
    if normalized_source_classes != normalized_target_classes:
        raise RuntimeError("strict model-state checkpoint class_to_idx mismatch")
    checkpoint_classifier_type = str(
        payload.get("config", {}).get("model", {}).get("classifier_type", "")
    ).strip().lower()
    normalized_expected_classifier = str(expected_classifier_type or "").strip().lower()
    if normalized_expected_classifier and (
        checkpoint_classifier_type != normalized_expected_classifier
    ):
        raise RuntimeError(
            "strict model-state checkpoint classifier_type mismatch: "
            f"expected={normalized_expected_classifier!r} "
            f"actual={checkpoint_classifier_type!r}"
        )

    source = payload["model"]
    target = model.state_dict()
    if expected_state_key_count is not None and len(source) != int(
        expected_state_key_count
    ):
        raise RuntimeError(
            "strict model-state source key count mismatch: "
            f"expected={int(expected_state_key_count)} actual={len(source)}"
        )
    if set(source) != set(target):
        raise RuntimeError(
            "strict model-state schema mismatch: "
            f"missing={sorted(set(target) - set(source))} "
            f"unexpected={sorted(set(source) - set(target))}"
        )
    incompatible: list[str] = []
    for name, incoming in source.items():
        current = target[name]
        if not isinstance(incoming, torch.Tensor) or not isinstance(
            current, torch.Tensor
        ):
            incompatible.append(name)
            continue
        if tuple(incoming.shape) != tuple(current.shape) or incoming.dtype != current.dtype:
            incompatible.append(name)
    if incompatible:
        raise RuntimeError(
            "strict model-state kind/shape/dtype mismatch: "
            f"{incompatible[:16]}"
        )

    model.load_state_dict(source, strict=True)
    loaded = model.state_dict()
    unequal = [
        name
        for name in source
        if not torch.equal(loaded[name].detach().cpu(), source[name].detach().cpu())
    ]
    if unequal:
        raise RuntimeError(
            "strict model-state post-load equality failed: " f"{unequal[:16]}"
        )
    loaded_keys = sorted(source)
    return {
        "mode": "strict_model_state_resume",
        "path": str(path.resolve()),
        "checkpoint_sha256": actual_digest,
        "expected_sha256": expected_digest,
        "source_state_keys": len(source),
        "target_state_keys": len(target),
        "loaded": len(source),
        "post_load_equal_keys": len(source),
        "loaded_keys": loaded_keys,
        "skipped": 0,
        "missing": 0,
        "unexpected": 0,
        "skipped_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "class_to_idx_verified": True,
        "classifier_type_verified": bool(normalized_expected_classifier),
        "checkpoint_classifier_type": checkpoint_classifier_type,
    }


def _reject_legacy_arcq_step_resume(model: nn.Module) -> None:
    """Forbid pretending any migrated ARCQ checkpoint is an exact resume.

    V1 has a different optimizer parameter topology because it contains dead
    terminal-H parameters. V2 also predates the FP32-H and smooth-observability
    v3 semantics. A user may load either payload as a weights-only
    initialization; restoring its old optimizer/scheduler/RNG state is not an
    exact continuation and therefore must be explicit rather than silent.
    """

    migrations = [
        module.last_load_migration
        for module in model.modules()
        if getattr(module, "last_load_migration", None) is not None
    ]
    if migrations:
        raise RuntimeError(
            "ARCQ weights were migrated to state v3, so this "
            "step checkpoint cannot resume its optimizer/scheduler/RNG state. "
            "Load it as a weights-only initialization and start a new optimizer. "
            f"Migration audit: {migrations}"
        )


def apply_trainable_prefixes(model: nn.Module, prefixes: list[str] | None) -> None:
    if not prefixes:
        return
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    matched = {prefix: 0 for prefix in prefixes}
    for name, param in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix):
                param.requires_grad_(True)
                matched[prefix] += int(param.numel())
    empty = [prefix for prefix, count in matched.items() if count == 0]
    if empty:
        raise ValueError(f"trainable prefixes matched no parameters: {empty}")
    print("Trainable prefixes:", matched)


def _classifier_family(model_cfg: dict[str, Any]) -> str:
    """Return the effective classifier family used by :func:`build_model`.

    Several public config spellings select the same implementation.  Treating
    those aliases as different would unnecessarily discard compatible
    ``backbone_kwargs`` when constructing a legacy, homogeneous teacher.
    """

    raw = str(
        model_cfg.get(
            "classifier_type",
            model_cfg.get("model_type", "c3"),
        )
    ).strip().lower()
    aliases = {
        "c3_farnet": "c3",
        "arcq_road_lite": "arcq_lite",
        "same_parameter_dualcnn": "dualcnn",
        "arcq_principle_control": "principle_control",
        "linear_backbone": "backbone_linear",
        "ceta_rspnet_released": "ceta_rspnet",
        "grit_rspnet_l_residual": "grit_rspnet_residual",
        "rspnet_grit_ledger": "rsp_grit_ledger",
        "rspnet_grit_spatial": "rsp_grit_spatial",
        "rsp_grit_spatial_transport": "rsp_grit_spatial",
        "rspnet_rcst_spatial": "rsp_rcst_spatial",
        "rsp_rcst_spatial_transport": "rsp_rcst_spatial",
        "rspnet_aort_spatial": "rsp_aort_spatial",
        "rsp_aort_spatial_transport": "rsp_aort_spatial",
    }
    return aliases.get(raw, raw)


def _teacher_backbone_kwargs(
    student_model_cfg: dict[str, Any],
    teacher_model_cfg: dict[str, Any],
    teacher_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the teacher's constructor-specific backbone options.

    ``backbone_kwargs`` is an atomic constructor argument bundle, not a nested
    set of generic model defaults.  Therefore an explicit teacher value
    replaces the student value in full.  Both ``null`` and ``{}`` explicitly
    mean "no teacher backbone options".  Without an explicit value, options
    are inherited only when teacher and student use the same classifier family
    *and* the same backbone.
    """

    if isinstance(teacher_overrides, dict) and "backbone_kwargs" in teacher_overrides:
        explicit = teacher_overrides["backbone_kwargs"]
        if explicit is None:
            return {}
        if not isinstance(explicit, dict):
            raise TypeError(
                "train.teacher_model_overrides.backbone_kwargs must be a "
                "mapping or null"
            )
        return copy.deepcopy(explicit)

    student_backbone = str(student_model_cfg.get("backbone", "")).strip().lower()
    teacher_backbone = str(teacher_model_cfg.get("backbone", "")).strip().lower()
    homogeneous = (
        _classifier_family(student_model_cfg) == _classifier_family(teacher_model_cfg)
        and student_backbone == teacher_backbone
    )
    if not homogeneous:
        return {}

    inherited = student_model_cfg.get("backbone_kwargs", {})
    if inherited is None:
        return {}
    if not isinstance(inherited, dict):
        # Preserve the existing public validation boundary and error wording.
        raise TypeError("model.backbone_kwargs must be a mapping")
    return copy.deepcopy(inherited)


def build_anchor_teacher(
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    device: torch.device,
) -> C3FaRNetSurfaceClassifier | None:
    train_cfg = cfg["train"]
    checkpoint = train_cfg.get("teacher_checkpoint")
    if not checkpoint:
        return None
    teacher_cfg = copy.deepcopy(cfg)
    student_model_cfg = cfg["model"]
    teacher_model_cfg = teacher_cfg["model"]
    teacher_model_cfg["backbone"] = str(train_cfg.get("teacher_backbone", "convnext_tiny"))
    teacher_model_cfg["head_type"] = str(train_cfg.get("teacher_head_type", "linear"))
    teacher_overrides = train_cfg.get("teacher_model_overrides")
    if isinstance(teacher_overrides, dict):
        # Resolve the constructor-specific option bundle separately.  A normal
        # recursive update would make an explicit ``backbone_kwargs: {}`` a
        # no-op and leak student-only arguments into an incompatible teacher.
        generic_overrides = {
            key: value
            for key, value in teacher_overrides.items()
            if key != "backbone_kwargs"
        }
        teacher_cfg["model"] = _deep_update_config(
            teacher_model_cfg,
            generic_overrides,
        )
        teacher_model_cfg = teacher_cfg["model"]
    teacher_model_cfg["backbone_kwargs"] = _teacher_backbone_kwargs(
        student_model_cfg,
        teacher_model_cfg,
        teacher_overrides if isinstance(teacher_overrides, dict) else None,
    )
    if not bool(train_cfg.get("teacher_preserve_hardpair_focus", False)):
        teacher_model_cfg["hardpair_focus_classes"] = []
        teacher_model_cfg["hardpair_focus_boundaries"] = []
    teacher = build_model(teacher_cfg, class_to_idx).to(device)
    load_audit = flexible_load(
        teacher,
        str(checkpoint),
        expected_sha256=train_cfg.get("teacher_checkpoint_sha256"),
    )
    if int(load_audit.get("loaded", 0) or 0) <= 0:
        raise RuntimeError(
            "anchor teacher checkpoint loaded zero model tensors; refusing a "
            f"random frozen teacher: {checkpoint}"
        )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"Frozen anchor teacher enabled: {checkpoint}")
    return teacher


def build_specialist_teacher(
    cfg: dict[str, Any],
    class_to_idx: dict[str, int],
    device: torch.device,
) -> nn.Module | None:
    train_cfg = cfg["train"]
    checkpoint = train_cfg.get("expert_teacher_checkpoint")
    if not checkpoint:
        return None
    teacher_cfg = copy.deepcopy(cfg)
    student_model_cfg = cfg["model"]
    teacher_model_cfg = teacher_cfg["model"]
    teacher_model_cfg["backbone"] = str(
        train_cfg.get("expert_teacher_backbone", train_cfg.get("teacher_backbone", teacher_model_cfg.get("backbone", "convnext_tiny")))
    )
    teacher_model_cfg["head_type"] = str(
        train_cfg.get("expert_teacher_head_type", train_cfg.get("teacher_head_type", teacher_model_cfg.get("head_type", "linear")))
    )
    expert_teacher_overrides = train_cfg.get("expert_teacher_model_overrides", train_cfg.get("teacher_model_overrides"))
    if isinstance(expert_teacher_overrides, dict):
        generic_overrides = {
            key: value
            for key, value in expert_teacher_overrides.items()
            if key != "backbone_kwargs"
        }
        teacher_cfg["model"] = _deep_update_config(
            teacher_model_cfg,
            generic_overrides,
        )
        teacher_model_cfg = teacher_cfg["model"]
    teacher_model_cfg["backbone_kwargs"] = _teacher_backbone_kwargs(
        student_model_cfg,
        teacher_model_cfg,
        (
            expert_teacher_overrides
            if isinstance(expert_teacher_overrides, dict)
            else None
        ),
    )
    teacher = build_model(teacher_cfg, class_to_idx).to(device)
    load_audit = flexible_load(
        teacher,
        str(checkpoint),
        expected_sha256=train_cfg.get("expert_teacher_checkpoint_sha256"),
    )
    if int(load_audit.get("loaded", 0) or 0) <= 0:
        raise RuntimeError(
            "specialist teacher checkpoint loaded zero model tensors; refusing "
            f"a random frozen teacher: {checkpoint}"
        )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    print(f"Frozen specialist teacher enabled: {checkpoint}")
    return teacher


def teacher_cache_key(path: str | os.PathLike[str]) -> str:
    """Stable key for cached teacher logits indexed by image path."""

    return os.path.normcase(os.path.abspath(str(path)))


_DEVELOPMENT_TEACHER_CACHE_ROLES = frozenset(
    {"train", "b0500", "val", "validation"}
)
_SHA256_HEX = frozenset("0123456789abcdef")


class TeacherLogitCache(dict[str, torch.Tensor]):
    """Path-indexed teacher targets with immutable audit metadata.

    This intentionally subclasses ``dict`` so every historical caller that
    supplied or consumed ``dict[path, logits]`` keeps working.  Strictly loaded
    caches additionally retain the per-path ground-truth label and the hashes
    that bind the cache, manifest, checkpoint, and provenance contract.
    """

    def __init__(
        self,
        rows: Mapping[str, torch.Tensor],
        *,
        labels_by_key: Mapping[str, int] | None = None,
        representation_rows: Mapping[
            str, Mapping[str, torch.Tensor]
        ] | None = None,
        representation_audits: Mapping[str, Mapping[str, Any]] | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(rows)
        self.labels_by_key = {
            str(key): int(value)
            for key, value in (labels_by_key or {}).items()
        }
        self.representation_rows = {
            str(name): dict(values)
            for name, values in (representation_rows or {}).items()
        }
        self.representation_audits = copy.deepcopy(
            dict(representation_audits or {})
        )
        self.audit = copy.deepcopy(dict(audit or {}))


def _normalized_sha256(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{field_name} must be exactly 64 hexadecimal characters")
    return text


def _teacher_cache_rows(cache: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    return cache


def load_teacher_logit_cache(
    path: str | os.PathLike[str] | None,
    *,
    require_contract: bool = False,
    expected_class_to_idx: dict[str, int] | None = None,
    expected_manifest_sha256: str | None = None,
    expected_image_size: int | None = None,
    expected_resize_mode: str | None = None,
    expected_augmentation: bool | None = None,
    expected_cache_sha256: str | None = None,
    expected_provenance_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_split_roles: Sequence[str] | None = None,
    allow_path_aligned_cross_view_expert: bool = False,
    logits_key: str = "logits",
    representation_key: str | None = None,
    cache_name: str | None = None,
) -> TeacherLogitCache | None:
    """Load an image-path -> logits cache created by `scripts/cache_teacher_logits.py`."""

    if not path:
        return None
    cache_path = Path(str(path))
    if not cache_path.exists():
        raise FileNotFoundError(f"teacher logits cache does not exist: {cache_path}")
    resolved_cache_path = cache_path.resolve()
    cache_sha256 = _sha256_file(resolved_cache_path)
    pinned_cache_sha256 = _normalized_sha256(
        expected_cache_sha256,
        field_name=f"{cache_name or 'teacher'} cache SHA256",
    )
    if pinned_cache_sha256 is not None and cache_sha256 != pinned_cache_sha256:
        raise ValueError(f"teacher logits cache file SHA256 mismatch: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"teacher logits cache payload must be a mapping: {cache_path}")
    image_paths = payload.get("image_paths")
    normalized_logits_key = str(logits_key).strip()
    if not normalized_logits_key:
        raise ValueError("teacher logits cache logits_key must be non-empty")
    logits = payload.get(normalized_logits_key)
    if image_paths is None or logits is None:
        raise ValueError(
            "teacher logits cache must contain image_paths and requested logits "
            f"key {normalized_logits_key!r}: {cache_path}"
        )
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError(
            f"teacher logits cache logits must be a rank-2 tensor: {cache_path}"
        )
    if not isinstance(image_paths, (list, tuple)):
        raise ValueError(
            f"teacher logits cache image_paths must be a list or tuple: {cache_path}"
        )
    if len(image_paths) != int(logits.shape[0]):
        raise ValueError(
            f"teacher logits cache length mismatch: paths={len(image_paths)} logits={tuple(logits.shape)} at {cache_path}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"teacher logits cache contains NaN or Inf: {cache_path}")
    normalized_keys = [teacher_cache_key(path_text) for path_text in image_paths]
    if len(normalized_keys) != len(set(normalized_keys)):
        raise ValueError(
            f"teacher logits cache contains duplicate normalized image paths: {cache_path}"
        )
    provenance = payload.get("provenance")
    provenance_sha256 = canonical_sha256(provenance)
    pinned_provenance_sha256 = _normalized_sha256(
        expected_provenance_sha256,
        field_name=f"{cache_name or 'teacher'} provenance SHA256",
    )
    if (
        pinned_provenance_sha256 is not None
        and provenance_sha256 != pinned_provenance_sha256
    ):
        raise ValueError(f"teacher logits cache provenance SHA256 mismatch: {cache_path}")

    cached_manifest_sha256 = _normalized_sha256(
        payload.get("manifest_sha256"),
        field_name=f"teacher logits cache manifest SHA256 at {cache_path}",
        required=require_contract,
    )
    manifest_audit = (
        provenance.get("manifest") if isinstance(provenance, dict) else None
    )
    provenance_manifest_sha256 = _normalized_sha256(
        manifest_audit.get("sha256") if isinstance(manifest_audit, dict) else None,
        field_name=f"teacher logits cache provenance manifest SHA256 at {cache_path}",
        required=require_contract,
    )
    if (
        cached_manifest_sha256 is not None
        and provenance_manifest_sha256 is not None
        and cached_manifest_sha256 != provenance_manifest_sha256
    ):
        raise ValueError(
            f"teacher logits cache top-level/provenance manifest SHA256 mismatch: {cache_path}"
        )
    pinned_manifest_sha256 = _normalized_sha256(
        expected_manifest_sha256,
        field_name="expected teacher-cache manifest SHA256",
    )
    if (
        pinned_manifest_sha256 is not None
        and cached_manifest_sha256 != pinned_manifest_sha256
    ):
        raise ValueError(f"teacher logits cache manifest SHA256 mismatch: {cache_path}")

    cached_checkpoint_sha256 = _normalized_sha256(
        payload.get("checkpoint_sha256"),
        field_name=f"teacher logits cache checkpoint SHA256 at {cache_path}",
        required=require_contract,
    )
    pinned_checkpoint_sha256 = _normalized_sha256(
        expected_checkpoint_sha256,
        field_name=f"{cache_name or 'teacher'} checkpoint SHA256",
    )
    if (
        pinned_checkpoint_sha256 is not None
        and cached_checkpoint_sha256 != pinned_checkpoint_sha256
    ):
        raise ValueError(f"teacher logits cache checkpoint SHA256 mismatch: {cache_path}")

    labels = payload.get("labels")
    labels_by_key: dict[str, int] = {}
    if labels is not None:
        if (
            not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or int(labels.shape[0]) != len(normalized_keys)
        ):
            raise ValueError(
                f"teacher logits cache labels must align with image_paths: {cache_path}"
            )
        labels = labels.long().cpu()
        if int(labels.numel()) and (
            int(labels.min()) < 0 or int(labels.max()) >= int(logits.shape[1])
        ):
            raise ValueError(
                f"teacher logits cache labels are outside the class range: {cache_path}"
            )
        labels_by_key = {
            key: int(labels[index]) for index, key in enumerate(normalized_keys)
        }
    elif require_contract:
        raise ValueError(f"strict teacher logits cache requires labels: {cache_path}")

    normalized_representation_key = str(representation_key or "").strip()
    representation_rows: dict[str, dict[str, torch.Tensor]] = {}
    representation_audits: dict[str, dict[str, Any]] = {}
    if normalized_representation_key:
        representations = payload.get("representations")
        if not isinstance(representations, Mapping):
            raise ValueError(
                "teacher cache representations mapping is missing for requested "
                f"key {normalized_representation_key!r}: {cache_path}"
            )
        representation = representations.get(normalized_representation_key)
        if (
            not isinstance(representation, torch.Tensor)
            or representation.ndim != 2
            or int(representation.shape[0]) != len(normalized_keys)
            or int(representation.shape[1]) <= 0
        ):
            raise ValueError(
                "teacher cached representation must be a non-empty rank-2 "
                "tensor aligned with image_paths: "
                f"key={normalized_representation_key!r} path={cache_path}"
            )
        if not bool(torch.isfinite(representation).all()):
            raise ValueError(
                "teacher cached representation contains NaN or Inf: "
                f"key={normalized_representation_key!r} path={cache_path}"
            )
        representation_contract = payload.get("representation_contract")
        if require_contract and not isinstance(representation_contract, Mapping):
            raise ValueError(
                f"strict teacher cache representation_contract is missing: {cache_path}"
            )
        if isinstance(representation_contract, Mapping):
            declared_dtype = str(representation_contract.get("dtype", "")).strip()
            if declared_dtype and declared_dtype != str(representation.dtype):
                raise ValueError(
                    "teacher cached representation dtype disagrees with its "
                    f"contract: declared={declared_dtype!r} "
                    f"actual={str(representation.dtype)!r} path={cache_path}"
                )
        representation = representation.detach().cpu()
        representation_rows[normalized_representation_key] = {
            key: representation[index]
            for index, key in enumerate(normalized_keys)
        }
        representation_audits[normalized_representation_key] = {
            "key": normalized_representation_key,
            "rows": int(representation.shape[0]),
            "width": int(representation.shape[1]),
            "dtype": str(representation.dtype),
            "contract": copy.deepcopy(representation_contract),
        }

    manifest_role = ""
    if isinstance(manifest_audit, dict):
        manifest_role = str(manifest_audit.get("role", "")).strip().lower()

    if require_contract:
        if int(payload.get("format_version", -1)) != 1 or int(
            payload.get("schema_version", -1)
        ) != 1:
            raise ValueError(
                f"teacher logits cache requires format/schema version 1: {cache_path}"
            )
        cached_class_to_idx = payload.get("class_to_idx")
        if expected_class_to_idx is None:
            raise ValueError(
                "expected_class_to_idx is required for strict teacher-cache loading"
            )
        normalized_expected_class_map = {
            str(name): int(index)
            for name, index in expected_class_to_idx.items()
        }
        if cached_class_to_idx != normalized_expected_class_map:
            raise ValueError(
                f"teacher logits cache class_to_idx mismatch: {cache_path}"
            )
        if int(logits.shape[1]) != len(normalized_expected_class_map):
            raise ValueError(
                "teacher logits cache class dimension mismatch: "
                f"logits={tuple(logits.shape)} classes={len(normalized_expected_class_map)}"
            )
        allowed_roles = {
            str(role).strip().lower()
            for role in (
                expected_split_roles
                if expected_split_roles is not None
                else _DEVELOPMENT_TEACHER_CACHE_ROLES
            )
        }
        if not allowed_roles or not allowed_roles.issubset(
            _DEVELOPMENT_TEACHER_CACHE_ROLES
        ):
            raise ValueError(
                "expected_split_roles must be a non-empty subset of development roles"
            )
        if manifest_role not in allowed_roles:
            raise ValueError(
                "formal-test guard rejected teacher-cache manifest role "
                f"{manifest_role!r}; expected one of {sorted(allowed_roles)}: {cache_path}"
            )
        transform_contract = payload.get("transform_contract")
        if not isinstance(transform_contract, dict):
            raise ValueError(
                f"teacher logits cache transform_contract is missing: {cache_path}"
            )
        if expected_image_size is not None and int(
            transform_contract.get("image_size", -1)
        ) != int(expected_image_size):
            raise ValueError(
                f"teacher logits cache image_size mismatch: {cache_path}"
            )
        if expected_resize_mode is not None and str(
            transform_contract.get("resize_mode", "")
        ).strip().lower() != str(expected_resize_mode).strip().lower():
            raise ValueError(
                f"teacher logits cache resize_mode mismatch: {cache_path}"
            )
        if expected_augmentation is not None and bool(
            transform_contract.get("augmentation", True)
        ) != bool(expected_augmentation):
            raise ValueError(
                f"teacher logits cache augmentation contract mismatch: {cache_path}"
            )
        if list(transform_contract.get("input_mean", [])) != [
            0.485,
            0.456,
            0.406,
        ] or list(transform_contract.get("input_std", [])) != [
            0.229,
            0.224,
            0.225,
        ]:
            raise ValueError(
                f"teacher logits cache normalization contract mismatch: {cache_path}"
            )
        usage_contract = (
            provenance.get("usage_contract")
            if isinstance(provenance, dict)
            else None
        )
        usage_scope = (
            str(usage_contract.get("scope", ""))
            if isinstance(usage_contract, dict)
            else ""
        )
        path_aligned_cross_view = (
            bool(allow_path_aligned_cross_view_expert)
            and usage_scope == "path_aligned_cross_view_expert"
        )
        if not isinstance(usage_contract, dict) or usage_scope not in {
            "deterministic_prewarm_only",
            "path_aligned_cross_view_expert",
        }:
            raise ValueError(
                f"teacher logits cache usage contract is missing: {cache_path}"
            )
        if path_aligned_cross_view:
            if not bool(usage_contract.get("path_and_label_alignment_required", False)):
                raise ValueError(
                    "cross-view expert cache must require path and label alignment: "
                    f"{cache_path}"
                )
        elif not bool(usage_contract.get("randomly_augmented_student_forbidden", False)):
            raise ValueError(
                f"teacher logits cache must forbid randomly augmented students: {cache_path}"
            )
        if not path_aligned_cross_view and not bool(
            usage_contract.get("student_view_must_match_protocol", False)
        ):
            raise ValueError(
                "teacher logits cache must declare whether the deterministic "
                f"student view matches its protocol: {cache_path}"
            )
        if not isinstance(provenance, dict) or bool(
            provenance.get("official_test_read", True)
        ):
            raise ValueError(
                f"teacher logits cache does not prove official-test isolation: {cache_path}"
            )
    logits = logits.float().cpu()
    rows = {
        normalized_keys[int(i)]: logits[int(i)]
        for i in range(len(normalized_keys))
    }
    audit = {
        "kind": "single_teacher_cache",
        "name": str(cache_name or cache_path.stem),
        "path": str(resolved_cache_path),
        "rows": len(rows),
        "classes": int(logits.shape[1]),
        "logits_key": normalized_logits_key,
        "cache_sha256": cache_sha256,
        "provenance_sha256": provenance_sha256,
        "manifest_sha256": cached_manifest_sha256,
        "manifest_role": manifest_role or None,
        "checkpoint_sha256": cached_checkpoint_sha256,
        "official_test_read": (
            bool(provenance.get("official_test_read"))
            if isinstance(provenance, dict)
            else None
        ),
        "transform_contract": copy.deepcopy(payload.get("transform_contract")),
        "usage_contract": copy.deepcopy(
            provenance.get("usage_contract")
            if isinstance(provenance, dict)
            else None
        ),
        "path_aligned_cross_view_expert": bool(
            allow_path_aligned_cross_view_expert
            and isinstance(provenance, dict)
            and isinstance(provenance.get("usage_contract"), dict)
            and provenance["usage_contract"].get("scope")
            == "path_aligned_cross_view_expert"
        ),
        "representations": copy.deepcopy(representation_audits),
    }
    cache = TeacherLogitCache(
        rows,
        labels_by_key=labels_by_key,
        representation_rows=representation_rows,
        representation_audits=representation_audits,
        audit=audit,
    )
    print(f"Loaded teacher logits cache: {cache_path} ({len(cache)} images)")
    return cache


def load_teacher_logit_ensemble_cache(
    cache_specs: Sequence[Mapping[str, Any]] | None,
    *,
    require_contract: bool,
    expected_class_to_idx: dict[str, int],
    expected_manifest_sha256: str,
    student_image_size: int,
    student_resize_mode: str,
    student_augmentation: bool,
    expected_split_roles: Sequence[str] = ("train", "b0500"),
    method: str = "equal_log_probability_mean",
    expected_ensemble_provenance_sha256: str | None = None,
    require_pinned_hashes: bool = False,
    allow_deterministic_cross_view: bool = False,
) -> TeacherLogitCache | None:
    """Load, align, and fuse arbitrary offline teachers without class routing.

    Each source is independently pinned and validated.  Alignment is by the
    normalized absolute image path, never by row position.  The only supported
    fusion is the fixed arithmetic mean of per-teacher log-softmax scores,
    matching the pre-registered S7/RSPNet validation diagnostic.
    """

    if cache_specs is None:
        return None
    if not isinstance(cache_specs, Sequence) or isinstance(cache_specs, (str, bytes)):
        raise TypeError("train.teacher_logits_caches must be a sequence of mappings")
    specs = list(cache_specs)
    if len(specs) < 2:
        raise ValueError("multi-teacher cache fusion requires at least two caches")
    normalized_method = str(method).strip().lower()
    if normalized_method != "equal_log_probability_mean":
        raise ValueError(
            "teacher-logit ensemble method must be equal_log_probability_mean"
        )
    normalized_student_resize_mode = str(student_resize_mode).strip().lower()
    if int(student_image_size) <= 0 or not normalized_student_resize_mode:
        raise ValueError("student image_size/resize_mode must be explicitly defined")
    if bool(student_augmentation):
        raise ValueError(
            "offline multi-teacher KD requires student augmentation=false"
        )

    loaded: list[TeacherLogitCache] = []
    names: set[str] = set()
    for index, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"teacher_logits_caches[{index}] must be a mapping")
        spec = dict(raw_spec)
        name = str(spec.get("name", f"teacher_{index}")).strip()
        if not name or name in names:
            raise ValueError(f"teacher cache names must be non-empty and unique: {name!r}")
        names.add(name)
        if require_pinned_hashes:
            for field_name in (
                "cache_sha256",
                "provenance_sha256",
                "checkpoint_sha256",
            ):
                _normalized_sha256(
                    spec.get(field_name),
                    field_name=f"teacher {name!r} {field_name}",
                    required=True,
                )
        source_manifest_sha = spec.get(
            "manifest_sha256", expected_manifest_sha256
        )
        if str(source_manifest_sha).strip().lower() != str(
            expected_manifest_sha256
        ).strip().lower():
            raise ValueError(
                f"teacher {name!r} manifest SHA256 differs from the training manifest"
            )
        loaded_cache = load_teacher_logit_cache(
            spec.get("path"),
            require_contract=require_contract,
            expected_class_to_idx=expected_class_to_idx,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_image_size=(
                int(spec["image_size"]) if spec.get("image_size") is not None else None
            ),
            expected_resize_mode=spec.get("resize_mode"),
            expected_augmentation=False,
            expected_cache_sha256=spec.get("cache_sha256"),
            expected_provenance_sha256=spec.get("provenance_sha256"),
            expected_checkpoint_sha256=spec.get("checkpoint_sha256"),
            expected_split_roles=expected_split_roles,
            logits_key=str(spec.get("logits_key", "logits")),
            representation_key=spec.get("representation_key"),
            cache_name=name,
        )
        if loaded_cache is None:
            raise ValueError(f"teacher {name!r} cache path is required")
        loaded.append(loaded_cache)

    student_transform = {
        "image_size": int(student_image_size),
        "resize_mode": normalized_student_resize_mode,
        "augmentation": False,
    }
    mismatched_source_views: list[dict[str, Any]] = []
    for source in loaded:
        source_transform = source.audit.get("transform_contract")
        if not isinstance(source_transform, Mapping):
            raise ValueError(
                f"teacher {source.audit.get('name')!r} lacks a transform contract"
            )
        normalized_source_transform = {
            "image_size": int(source_transform.get("image_size", -1)),
            "resize_mode": str(source_transform.get("resize_mode", ""))
            .strip()
            .lower(),
            "augmentation": bool(source_transform.get("augmentation", True)),
        }
        if normalized_source_transform != student_transform:
            mismatched_source_views.append(
                {
                    "name": str(source.audit.get("name")),
                    "teacher_transform": normalized_source_transform,
                    "student_transform": dict(student_transform),
                    "source_contract_student_view_must_match_protocol": bool(
                        (
                            source.audit.get("usage_contract")
                            if isinstance(source.audit.get("usage_contract"), Mapping)
                            else {}
                        ).get("student_view_must_match_protocol", False)
                    ),
                }
            )
    if mismatched_source_views and not bool(allow_deterministic_cross_view):
        mismatch_names = [item["name"] for item in mismatched_source_views]
        raise ValueError(
            "teacher/student deterministic view mismatch for sources "
            f"{mismatch_names}; set "
            "teacher_logits_ensemble.allow_deterministic_cross_view=true "
            "to acknowledge this cross-view KD override"
        )

    reference_keys = set(loaded[0])
    reference_labels = loaded[0].labels_by_key
    if not reference_labels:
        raise ValueError("multi-teacher strict alignment requires cached labels")
    for source in loaded[1:]:
        source_keys = set(source)
        if source_keys != reference_keys:
            only_reference = sorted(reference_keys - source_keys)[:3]
            only_source = sorted(source_keys - reference_keys)[:3]
            raise ValueError(
                "teacher cache path sets differ after normalization: "
                f"only_reference={only_reference} only_source={only_source}"
            )
        disagreements = [
            key
            for key in reference_keys
            if source.labels_by_key.get(key) != reference_labels.get(key)
        ]
        if disagreements:
            raise ValueError(
                "teacher cache labels disagree after image_path alignment; "
                f"first={disagreements[:3]}"
            )

    rows: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        source_log_probabilities = torch.stack(
            [F.log_softmax(source[key].float(), dim=-1) for source in loaded],
            dim=0,
        )
        rows[key] = source_log_probabilities.mean(dim=0).contiguous()

    source_audits = [copy.deepcopy(source.audit) for source in loaded]
    # Representation loading must not mutate the already pinned provenance of
    # the fixed logit ensemble.  The same source file hashes bind the optional
    # representation, while its own details live in a separate runtime audit.
    logit_source_audits = []
    for source_audit in source_audits:
        source_logit_audit = copy.deepcopy(source_audit)
        source_logit_audit.pop("representations", None)
        logit_source_audits.append(source_logit_audit)
    representation_rows: dict[str, dict[str, torch.Tensor]] = {}
    representation_audits: dict[str, dict[str, Any]] = {}
    for source in loaded:
        for representation_key, source_rows in source.representation_rows.items():
            if representation_key in representation_rows:
                raise ValueError(
                    "multiple teacher sources requested the same representation "
                    f"key {representation_key!r}; the relation target would be ambiguous"
                )
            if set(source_rows) != reference_keys:
                raise ValueError(
                    "teacher representation path set differs from aligned logits: "
                    f"key={representation_key!r}"
                )
            representation_rows[representation_key] = dict(source_rows)
            representation_audits[representation_key] = {
                **copy.deepcopy(
                    source.representation_audits.get(representation_key, {})
                ),
                "source_name": str(source.audit.get("name")),
                "source_cache_sha256": str(source.audit.get("cache_sha256")),
                "source_provenance_sha256": str(
                    source.audit.get("provenance_sha256")
                ),
                "source_checkpoint_sha256": str(
                    source.audit.get("checkpoint_sha256")
                ),
            }
    ensemble_provenance = {
        "schema_version": 1,
        "kind": "fixed_offline_teacher_ensemble",
        "fusion": {
            "method": normalized_method,
            "teacher_count": len(loaded),
            "weights": [1.0 / float(len(loaded))] * len(loaded),
            "formula": "mean(log_softmax(source_logits), dim=teachers)",
            "class_or_sample_routing": False,
        },
        "alignment": {
            "key": "normcase(abspath(image_path))",
            "exact_path_set_equality": True,
            "exact_per_path_label_equality": True,
        },
        "manifest_sha256": str(expected_manifest_sha256).strip().lower(),
        "manifest_roles": sorted(
            {str(source.audit.get("manifest_role")) for source in loaded}
        ),
        "student_transform": student_transform,
        "mismatched_source_views": mismatched_source_views,
        "allow_deterministic_cross_view": bool(
            allow_deterministic_cross_view
        ),
        "cross_view_override": bool(mismatched_source_views),
        "sources": logit_source_audits,
        "official_test_read": False,
    }
    ensemble_provenance_sha256 = canonical_sha256(ensemble_provenance)
    pinned_ensemble_sha256 = _normalized_sha256(
        expected_ensemble_provenance_sha256,
        field_name="teacher-logit ensemble provenance SHA256",
        required=require_pinned_hashes,
    )
    if (
        pinned_ensemble_sha256 is not None
        and ensemble_provenance_sha256 != pinned_ensemble_sha256
    ):
        raise ValueError("teacher-logit ensemble provenance SHA256 mismatch")
    audit = {
        "kind": "fixed_offline_teacher_ensemble",
        "rows": len(rows),
        "classes": len(expected_class_to_idx),
        "manifest_sha256": str(expected_manifest_sha256).strip().lower(),
        "source_count": len(loaded),
        "source_names": [str(source.audit.get("name")) for source in loaded],
        "source_cache_sha256": [
            str(source.audit.get("cache_sha256")) for source in loaded
        ],
        "source_provenance_sha256": [
            str(source.audit.get("provenance_sha256")) for source in loaded
        ],
        "source_checkpoint_sha256": [
            str(source.audit.get("checkpoint_sha256")) for source in loaded
        ],
        "student_transform": student_transform,
        "mismatched_source_views": mismatched_source_views,
        "cross_view_override": bool(mismatched_source_views),
        "ensemble_provenance_sha256": ensemble_provenance_sha256,
        "ensemble_provenance": ensemble_provenance,
        "official_test_read": False,
        "representations": copy.deepcopy(representation_audits),
    }
    print(
        "Loaded fixed equal-log-probability teacher ensemble: "
        f"teachers={len(loaded)} images={len(rows)} "
        f"provenance_sha256={ensemble_provenance_sha256}"
    )
    return TeacherLogitCache(
        rows,
        labels_by_key=reference_labels,
        representation_rows=representation_rows,
        representation_audits=representation_audits,
        audit=audit,
    )


def _audit_single_teacher_student_view(
    cache: TeacherLogitCache | None,
    *,
    data_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    cache_contract: Mapping[str, Any],
) -> None:
    """Fail closed on an unacknowledged deterministic cross-view cache.

    A cached teacher describes its own image transform, not the student's.
    Historically both were the same scalar square view.  Rectangular students
    make that assumption false, so the override must be explicit and recorded
    rather than achieved by rewriting the pinned teacher metadata.
    """

    if cache is None:
        return
    teacher_transform_raw = cache.audit.get("transform_contract")
    if not isinstance(teacher_transform_raw, Mapping):
        raise ValueError("single teacher cache lacks a transform contract")
    teacher_image_size = int(teacher_transform_raw.get("image_size", -1))
    if teacher_image_size <= 0:
        raise ValueError("single teacher cache has an invalid image_size contract")
    teacher_input_hw_raw = teacher_transform_raw.get("input_hw")
    if teacher_input_hw_raw is None:
        teacher_input_hw = (teacher_image_size, teacher_image_size)
    elif (
        isinstance(teacher_input_hw_raw, Sequence)
        and not isinstance(teacher_input_hw_raw, (str, bytes))
        and len(teacher_input_hw_raw) == 2
    ):
        teacher_input_hw = (
            int(teacher_input_hw_raw[0]),
            int(teacher_input_hw_raw[1]),
        )
    else:
        raise ValueError("single teacher cache input_hw must contain (height, width)")
    _, student_input_hw = _resolve_data_input_geometry(data_cfg)
    teacher_transform = {
        "input_hw": [int(teacher_input_hw[0]), int(teacher_input_hw[1])],
        "image_size": teacher_image_size,
        "resize_mode": str(teacher_transform_raw.get("resize_mode", ""))
        .strip()
        .lower(),
        "augmentation": bool(teacher_transform_raw.get("augmentation", True)),
    }
    student_transform = {
        "input_hw": [int(student_input_hw[0]), int(student_input_hw[1])],
        "image_size": int(data_cfg.get("image_size", 192)),
        "resize_mode": str(data_cfg.get("train_resize_mode", "letterbox"))
        .strip()
        .lower(),
        "augmentation": bool(train_cfg.get("augmentation", True)),
    }
    mismatch = (
        teacher_transform["input_hw"] != student_transform["input_hw"]
        or teacher_transform["resize_mode"] != student_transform["resize_mode"]
        or teacher_transform["augmentation"] != student_transform["augmentation"]
    )
    allow_cross_view = bool(
        cache_contract.get("allow_deterministic_cross_view", False)
    )
    if mismatch and student_transform["augmentation"]:
        raise ValueError(
            "teacher/student cross-view KD requires a deterministic student view"
        )
    if mismatch and not allow_cross_view:
        raise ValueError(
            "single teacher/student deterministic view mismatch; set "
            "train.teacher_logits_cache_contract."
            "allow_deterministic_cross_view=true to acknowledge the override"
        )
    cache.audit["student_transform"] = student_transform
    cache.audit["teacher_transform_normalized"] = teacher_transform
    cache.audit["allow_deterministic_cross_view"] = allow_cross_view
    cache.audit["cross_view_override"] = bool(mismatch)


def cached_teacher_logits_for_batch(
    cache: Mapping[str, torch.Tensor] | None,
    image_paths: Sequence[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
    cache_name: str,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if cache is None:
        return None, {}
    rows: list[torch.Tensor] = []
    missing: list[str] = []
    label_mismatches: list[str] = []
    cached_labels = (
        cache.labels_by_key if isinstance(cache, TeacherLogitCache) else {}
    )
    if labels is not None and int(labels.numel()) != len(image_paths):
        raise ValueError("batch labels must align with image_paths")
    batch_labels = labels.detach().cpu().tolist() if labels is not None else None
    for index, path_text in enumerate(image_paths):
        key = teacher_cache_key(path_text)
        row = cache.get(key)
        if row is None:
            missing.append(str(path_text))
        else:
            rows.append(row)
        if batch_labels is not None and cached_labels:
            cached_label = cached_labels.get(key)
            if cached_label is None or int(cached_label) != int(batch_labels[index]):
                label_mismatches.append(str(path_text))
    if missing:
        if strict:
            preview = ", ".join(missing[:3])
            raise KeyError(f"{cache_name} teacher logits cache missing {len(missing)} paths, first: {preview}")
        return None, {f"{cache_name}_teacher_cache_miss": float(len(missing))}
    if label_mismatches:
        preview = ", ".join(label_mismatches[:3])
        raise ValueError(
            f"{cache_name} teacher cache label mismatch for "
            f"{len(label_mismatches)} paths, first: {preview}"
        )
    logits = torch.stack(rows).to(device=device, dtype=dtype, non_blocking=True)
    logs = {f"{cache_name}_teacher_cache_hit": float(len(rows))}
    if batch_labels is not None and cached_labels:
        logs[f"{cache_name}_teacher_cache_label_verified"] = float(len(rows))
    return logits, logs


def cached_teacher_representation_for_batch(
    cache: Mapping[str, torch.Tensor] | None,
    image_paths: Sequence[str],
    *,
    representation_key: str,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
    cache_name: str,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Read one audited offline representation by normalized path and label.

    Representations are deliberately attached to ``TeacherLogitCache`` rather
    than loaded by a second online teacher path.  This keeps the same file,
    checkpoint, manifest, provenance and formal-test guards that protected the
    cached logits.
    """

    normalized_key = str(representation_key).strip()
    if not normalized_key:
        raise ValueError("teacher representation_key must be non-empty")
    if not isinstance(cache, TeacherLogitCache):
        if strict:
            raise TypeError(
                "strict cached representation distillation requires an audited "
                "TeacherLogitCache"
            )
        return None, {f"{cache_name}_teacher_representation_unavailable": 1.0}
    source_rows = cache.representation_rows.get(normalized_key)
    if source_rows is None:
        if strict:
            raise KeyError(
                f"{cache_name} teacher cache lacks representation "
                f"{normalized_key!r}"
            )
        return None, {f"{cache_name}_teacher_representation_unavailable": 1.0}
    if labels is not None and int(labels.numel()) != len(image_paths):
        raise ValueError("batch labels must align with image_paths")
    batch_labels = labels.detach().cpu().tolist() if labels is not None else None
    rows: list[torch.Tensor] = []
    missing: list[str] = []
    label_mismatches: list[str] = []
    for index, path_text in enumerate(image_paths):
        path_key = teacher_cache_key(path_text)
        row = source_rows.get(path_key)
        if row is None:
            missing.append(str(path_text))
        else:
            rows.append(row)
        if batch_labels is not None:
            cached_label = cache.labels_by_key.get(path_key)
            if cached_label is None or int(cached_label) != int(batch_labels[index]):
                label_mismatches.append(str(path_text))
    if missing:
        if strict:
            preview = ", ".join(missing[:3])
            raise KeyError(
                f"{cache_name} teacher representation cache missing "
                f"{len(missing)} paths, first: {preview}"
            )
        return None, {
            f"{cache_name}_teacher_representation_cache_miss": float(len(missing))
        }
    if label_mismatches:
        preview = ", ".join(label_mismatches[:3])
        raise ValueError(
            f"{cache_name} teacher representation label mismatch for "
            f"{len(label_mismatches)} paths, first: {preview}"
        )
    representation = torch.stack(rows).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    if representation.ndim != 2 or not bool(torch.isfinite(representation).all()):
        raise ValueError(
            f"{cache_name} teacher representation batch is invalid"
        )
    logs = {
        f"{cache_name}_teacher_representation_cache_hit": float(len(rows)),
        f"{cache_name}_teacher_representation_width": float(
            representation.shape[1]
        ),
    }
    if batch_labels is not None:
        logs[f"{cache_name}_teacher_representation_label_verified"] = float(
            len(rows)
        )
    return representation, logs


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _training_step_checkpoint_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    cfg: dict[str, Any],
    epoch: int,
    step: int,
    total_steps: int,
    class_to_idx: dict[str, int] | None,
    run_provenance: dict[str, Any] | None,
    train_partial: dict[str, Any],
    ema: ModelEMA | None,
) -> dict[str, Any]:
    """Build the single step-checkpoint schema shared by every train loop.

    Step checkpoints are written only at optimizer-update boundaries, so there
    are no pending gradients to serialize.  Keeping construction centralized
    prevents a fast path from silently omitting exact-resume state.
    """

    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scaler_enabled": bool(scaler.is_enabled()),
        "epoch": int(epoch),
        "step": int(step),
        "total_steps": int(total_steps),
        "train_dataset_size": int(len(loader.dataset)),
        "train_batch_size": int(
            getattr(loader, "batch_size", None)
            or getattr(getattr(loader, "batch_sampler", None), "batch_size", 0)
            or 0
        ),
        "class_to_idx": class_to_idx or {},
        "config": cfg,
        "load_audit": dict(cfg.get("train", {}).get("_load_audit", {}) or {}),
        "provenance": copy.deepcopy(run_provenance or {}),
        "train_partial": train_partial,
        "rng_state": _capture_rng_state(),
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    return state


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def anchor_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = float(loss_cfg.get("anchor_consistency_weight", 0.0))
    focus_weight = float(loss_cfg.get("anchor_consistency_focus_weight", 0.0))
    if weight <= 0.0 and focus_weight <= 0.0:
        return student_logits.new_zeros(()), {"loss_anchor_consistency": 0.0, "anchor_consistency_count": 0.0}
    temperature = max(float(loss_cfg.get("anchor_consistency_temperature", 2.0)), 1e-3)
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        s = student_logits.float() / temperature
        t = teacher_logits.float() / temperature
        per_sample = F.kl_div(
            F.log_softmax(s, dim=1),
            F.softmax(t, dim=1),
            reduction="none",
        ).sum(dim=1) * (temperature * temperature)
        teacher_prob = F.softmax(teacher_logits.float(), dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_margin = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_margin = torch.ones_like(teacher_conf)
        teacher_pred = teacher_top2.indices[:, 0]
        teacher_correct = teacher_pred.eq(labels)
    focus_classes = {canonical_class_label(name) for name in loss_cfg.get("anchor_consistency_exempt_classes", [])}
    if focus_classes:
        focus_idx = {
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in focus_classes
        }
        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in focus_idx:
            focus_mask |= labels.eq(int(idx))
    else:
        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    nonfocus_mask = ~focus_mask
    terms: list[torch.Tensor] = []
    weighted_count = 0.0
    if weight > 0.0 and bool(nonfocus_mask.any()):
        terms.append(float(weight) * per_sample[nonfocus_mask].mean())
        weighted_count += float(nonfocus_mask.sum().detach().cpu())
    if focus_weight > 0.0 and bool(focus_mask.any()):
        focus_loss_mask = focus_mask
        low_margin_threshold = float(loss_cfg.get("anchor_consistency_focus_low_margin_threshold", -1.0))
        if low_margin_threshold >= 0.0:
            focus_loss_mask = focus_loss_mask & teacher_margin.le(low_margin_threshold)
        if bool(focus_loss_mask.any()):
            terms.append(float(focus_weight) * per_sample[focus_loss_mask].mean())
            weighted_count += float(focus_loss_mask.sum().detach().cpu())
    protect_weight = float(loss_cfg.get("anchor_consistency_protect_weight", 0.0))
    protect_conf = float(loss_cfg.get("anchor_consistency_protect_confidence", 0.0))
    protect_margin = float(loss_cfg.get("anchor_consistency_protect_margin", 0.0))
    protect_mask = teacher_correct
    if protect_conf > 0.0:
        protect_mask = protect_mask & teacher_conf.ge(protect_conf)
    if protect_margin > 0.0:
        protect_mask = protect_mask & teacher_margin.ge(protect_margin)
    if protect_weight > 0.0 and bool(protect_mask.any()):
        terms.append(float(protect_weight) * per_sample[protect_mask].mean())
        weighted_count += float(protect_mask.sum().detach().cpu())
    no_flip_weight = float(loss_cfg.get("anchor_no_flip_weight", 0.0))
    no_flip_mask = protect_mask
    if bool(loss_cfg.get("anchor_no_flip_nonfocus_only", True)):
        no_flip_mask = no_flip_mask & nonfocus_mask
    if no_flip_weight > 0.0 and bool(no_flip_mask.any()):
        no_flip = F.cross_entropy(student_logits.float().index_select(0, no_flip_mask.nonzero(as_tuple=False).flatten()), teacher_pred.index_select(0, no_flip_mask.nonzero(as_tuple=False).flatten()))
        terms.append(float(no_flip_weight) * no_flip.to(dtype=student_logits.dtype))
    if not terms:
        return student_logits.new_zeros(()), {"loss_anchor_consistency": 0.0, "anchor_consistency_count": 0.0}
    loss = torch.stack(terms).sum().to(dtype=student_logits.dtype)
    return loss, {
        "loss_anchor_consistency": float(loss.detach().cpu()),
        "anchor_consistency_count": weighted_count,
        "anchor_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "anchor_teacher_margin_mean": float(teacher_margin.detach().mean().cpu()),
        "anchor_protect_count": float(protect_mask.sum().detach().cpu()),
        "anchor_no_flip_count": float(no_flip_mask.sum().detach().cpu()),
    }


def anchor_nonregression_barrier_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Class-balanced anchor barrier for RSCD negative-transfer control.

    This is a task-adapted GEM/LwF-style constraint. Focused RSCD mechanisms may
    update weak concrete/water boundaries, but samples that the anchor teacher
    already classifies correctly should not receive a higher true-label CE loss
    beyond a small tolerance. The penalty is averaged per class first, so large
    asphalt/snow groups cannot hide regressions in smaller hard classes.
    """

    weight = float(loss_cfg.get("anchor_nonregression_weight", 0.0))
    focus_weight = float(loss_cfg.get("anchor_nonregression_focus_weight", weight))
    if weight <= 0.0 and focus_weight <= 0.0:
        return student_logits.new_zeros(()), {
            "loss_anchor_nonregression": 0.0,
            "anchor_nonregression_count": 0.0,
        }
    tolerance = max(float(loss_cfg.get("anchor_nonregression_margin", 0.0)), 0.0)
    protect_conf = float(loss_cfg.get("anchor_nonregression_confidence", 0.0))
    protect_margin = float(loss_cfg.get("anchor_nonregression_teacher_margin", 0.0))
    use_squared = bool(loss_cfg.get("anchor_nonregression_squared", True))
    focus_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("anchor_nonregression_focus_classes", loss_cfg.get("focus_ce_classes", []))
    }
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in focus_idx:
        focus_mask |= labels.eq(int(idx))
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        student_ce = F.cross_entropy(student_logits.float(), labels, reduction="none")
        teacher_ce = F.cross_entropy(teacher_logits.float(), labels, reduction="none")
        teacher_prob = F.softmax(teacher_logits.float(), dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_gap = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_gap = torch.ones_like(teacher_conf)
        teacher_pred = teacher_top2.indices[:, 0]
        protect_mask = teacher_pred.eq(labels)
        if protect_conf > 0.0:
            protect_mask = protect_mask & teacher_conf.ge(protect_conf)
        if protect_margin > 0.0:
            protect_mask = protect_mask & teacher_gap.ge(protect_margin)
        excess = F.relu(student_ce - teacher_ce.detach() - tolerance)
        if use_squared:
            excess = excess.pow(2)
    terms: list[torch.Tensor] = []
    weighted_count = 0.0
    active_classes = 0
    active_excess_sum = 0.0
    for class_idx in labels.detach().unique().tolist():
        class_mask = labels.eq(int(class_idx)) & protect_mask
        if not bool(class_mask.any()):
            continue
        class_is_focus = int(class_idx) in focus_idx
        class_weight = focus_weight if class_is_focus else weight
        if class_weight <= 0.0:
            continue
        class_loss = excess[class_mask].mean()
        terms.append(float(class_weight) * class_loss)
        active_classes += 1
        weighted_count += float(class_mask.sum().detach().cpu())
        active_excess_sum += float(excess[class_mask].detach().mean().cpu())
    if not terms:
        return student_logits.new_zeros(()), {
            "loss_anchor_nonregression": 0.0,
            "anchor_nonregression_count": 0.0,
            "anchor_nonregression_active_classes": 0.0,
        }
    loss = torch.stack(terms).mean().to(dtype=student_logits.dtype)
    return loss, {
        "loss_anchor_nonregression": float(loss.detach().cpu()),
        "anchor_nonregression_count": weighted_count,
        "anchor_nonregression_active_classes": float(active_classes),
        "anchor_nonregression_excess_mean": active_excess_sum / max(active_classes, 1),
        "anchor_nonregression_protect_rate": float(protect_mask.float().detach().mean().cpu()),
        "anchor_nonregression_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "anchor_nonregression_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
    }


def pareto_safe_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Classwise no-harm distillation for RSCD hard-boundary fine tuning.

    This loss is a task-adapted LwF/GEM-style guard. It lets the new RSCD
    hard-class mechanism optimize weak wet/water/concrete boundaries, but it
    keeps teacher-correct samples from losing either their full probability
    distribution or their true-vs-nearest-boundary margin. The averaging is
    class-balanced so a gain on one large class cannot hide regressions on a
    smaller protected class.
    """

    weight = float(loss_cfg.get("pareto_safe_distill_weight", 0.0))
    margin_weight = float(loss_cfg.get("pareto_safe_margin_weight", 0.0))
    hardpair_weight = float(loss_cfg.get("pareto_safe_hardpair_margin_weight", 0.0))
    if weight <= 0.0 and margin_weight <= 0.0 and hardpair_weight <= 0.0:
        return student_logits.new_zeros(()), {
            "loss_pareto_safe_distill": 0.0,
            "pareto_safe_protect_count": 0.0,
            "pareto_safe_hardpair_count": 0.0,
        }

    temperature = max(float(loss_cfg.get("pareto_safe_temperature", 2.0)), 1e-3)
    protect_conf = float(loss_cfg.get("pareto_safe_confidence", 0.0))
    protect_margin = float(loss_cfg.get("pareto_safe_teacher_margin", 0.0))
    tolerance = max(float(loss_cfg.get("pareto_safe_margin_tolerance", 0.0)), 0.0)
    focus_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("pareto_safe_focus_classes", loss_cfg.get("focus_ce_classes", []))
    }
    protected_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("pareto_safe_protected_classes", [])
    }
    exempt_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("pareto_safe_exempt_classes", [])
    }
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    protected_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in protected_classes
    }
    exempt_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in exempt_classes
    }

    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        s_logits = student_logits.float()
        t_logits = teacher_logits.float()
        teacher_prob = F.softmax(t_logits, dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_pred = teacher_top2.indices[:, 0]
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_gap = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_gap = torch.ones_like(teacher_conf)
        protect_mask = teacher_pred.eq(labels)
        if protect_conf > 0.0:
            protect_mask = protect_mask & teacher_conf.ge(protect_conf)
        if protect_margin > 0.0:
            protect_mask = protect_mask & teacher_gap.ge(protect_margin)
        if protected_idx:
            class_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in protected_idx:
                class_mask |= labels.eq(int(idx))
            protect_mask = protect_mask & class_mask
        if exempt_idx:
            exempt_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in exempt_idx:
                exempt_mask |= labels.eq(int(idx))
            protect_mask = protect_mask & ~exempt_mask

        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in focus_idx:
            focus_mask |= labels.eq(int(idx))
        focus_protect_scale = float(loss_cfg.get("pareto_safe_focus_protect_scale", 0.25))
        focus_protect_scale = min(max(focus_protect_scale, 0.0), 1.0)

        kl_values = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)

        true_student = s_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        true_teacher = t_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        inf = torch.finfo(s_logits.dtype).max
        one_hot = F.one_hot(labels, num_classes=s_logits.size(1)).bool()
        student_other = s_logits.masked_fill(one_hot, -inf).max(dim=1).values
        teacher_other = t_logits.masked_fill(one_hot, -inf).max(dim=1).values
        student_true_margin = true_student - student_other
        teacher_true_margin = true_teacher - teacher_other
        margin_barrier = F.relu(teacher_true_margin.detach() - tolerance - student_true_margin)
        if bool(loss_cfg.get("pareto_safe_margin_squared", True)):
            margin_barrier = margin_barrier.pow(2)

    terms: list[torch.Tensor] = []
    active_classes = 0
    protect_count = 0.0
    kl_sum = 0.0
    margin_sum = 0.0
    for class_idx in labels.detach().unique().tolist():
        class_idx = int(class_idx)
        class_mask = labels.eq(class_idx) & protect_mask
        if not bool(class_mask.any()):
            continue
        class_scale = focus_protect_scale if class_idx in focus_idx else 1.0
        if class_scale <= 0.0:
            continue
        class_terms: list[torch.Tensor] = []
        if weight > 0.0:
            class_kl = kl_values[class_mask].mean()
            class_terms.append(float(weight) * class_kl)
            kl_sum += float(class_kl.detach().cpu())
        if margin_weight > 0.0:
            class_margin = margin_barrier[class_mask].mean()
            class_terms.append(float(margin_weight) * class_margin)
            margin_sum += float(class_margin.detach().cpu())
        if class_terms:
            terms.append(float(class_scale) * torch.stack(class_terms).sum())
            active_classes += 1
            protect_count += float(class_mask.sum().detach().cpu())

    hardpair_terms: list[torch.Tensor] = []
    hardpair_count = 0
    hardpair_violation_sum = 0.0
    if hardpair_weight > 0.0:
        idx_to_name = {int(idx): canonical_class_label(name) for idx, name in idx_to_class.items()}
        requested_pairs: set[frozenset[str]] = set()
        for item in loss_cfg.get("pareto_safe_hardpair_pairs", []):
            parts = str(item).replace("<->", "|").replace(",", "|").split("|")
            if len(parts) == 2:
                requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            if requested_pairs:
                names = frozenset((idx_to_name.get(left, str(left)), idx_to_name.get(right, str(right))))
                if names not in requested_pairs:
                    continue
            pair_mask = (labels.eq(left) | labels.eq(right)) & protect_mask
            if not bool(pair_mask.any()):
                continue
            sign = torch.where(labels.eq(left), 1.0, -1.0).to(device=s_logits.device, dtype=s_logits.dtype)
            teacher_pair_margin = sign * (t_logits[:, left] - t_logits[:, right])
            student_pair_margin = sign * (s_logits[:, left] - s_logits[:, right])
            violation = F.relu(teacher_pair_margin.detach() - tolerance - student_pair_margin)
            if bool(loss_cfg.get("pareto_safe_margin_squared", True)):
                violation = violation.pow(2)
            selected = violation[pair_mask]
            if bool(selected.numel()):
                hardpair_terms.append(selected.mean())
                hardpair_count += int(pair_mask.sum().detach().cpu())
                hardpair_violation_sum += float(selected.detach().mean().cpu())

    if hardpair_terms:
        terms.append(float(hardpair_weight) * torch.stack(hardpair_terms).mean())
    if not terms:
        return student_logits.new_zeros(()), {
            "loss_pareto_safe_distill": 0.0,
            "pareto_safe_protect_count": 0.0,
            "pareto_safe_hardpair_count": 0.0,
            "pareto_safe_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
            "pareto_safe_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
        }
    loss = torch.stack(terms).mean().to(dtype=student_logits.dtype)
    return loss, {
        "loss_pareto_safe_distill": float(loss.detach().cpu()),
        "pareto_safe_protect_count": protect_count,
        "pareto_safe_active_classes": float(active_classes),
        "pareto_safe_hardpair_count": float(hardpair_count),
        "pareto_safe_kl_mean": kl_sum / max(active_classes, 1),
        "pareto_safe_margin_excess_mean": margin_sum / max(active_classes, 1),
        "pareto_safe_hardpair_excess_mean": hardpair_violation_sum / max(len(hardpair_terms), 1),
        "pareto_safe_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "pareto_safe_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
        "pareto_safe_focus_count": float(focus_mask.sum().detach().cpu()),
    }


def dual_teacher_noharm_loss(
    student_logits: torch.Tensor,
    anchor_logits: torch.Tensor | None,
    expert_logits: torch.Tensor | None,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single-student distillation that separates weak-class repair from no-harm protection.

    RSCD improvements often help wet/water/concrete/slight boundaries while
    hurting already-stable classes. This loss routes each training sample to one
    frozen teacher: a specialist teacher for explicitly improved weak classes,
    and the anchor teacher for protected or non-focus classes. Both routes are
    class-balanced and gated by teacher correctness/confidence, so a local repair
    cannot silently buy gains by damaging another RSCD composite class.
    """

    total_weight = float(loss_cfg.get("dual_teacher_noharm_weight", 0.0))
    expert_weight = float(loss_cfg.get("dual_teacher_expert_weight", 1.0))
    anchor_weight = float(loss_cfg.get("dual_teacher_anchor_weight", 1.0))
    if total_weight <= 0.0 or (expert_logits is None and anchor_logits is None):
        return student_logits.new_zeros(()), {
            "loss_dual_teacher_noharm": 0.0,
            "dual_teacher_expert_count": 0.0,
            "dual_teacher_anchor_count": 0.0,
        }

    temperature = max(float(loss_cfg.get("dual_teacher_temperature", 2.0)), 1e-3)
    expert_conf = float(loss_cfg.get("dual_teacher_expert_confidence", 0.0))
    expert_margin = float(loss_cfg.get("dual_teacher_expert_margin", 0.0))
    anchor_conf = float(loss_cfg.get("dual_teacher_anchor_confidence", 0.0))
    anchor_margin = float(loss_cfg.get("dual_teacher_anchor_margin", 0.0))
    margin_weight = float(loss_cfg.get("dual_teacher_margin_weight", 0.0))
    margin_tolerance = max(float(loss_cfg.get("dual_teacher_margin_tolerance", 0.0)), 0.0)
    require_teacher_correct = bool(loss_cfg.get("dual_teacher_require_teacher_correct", True))
    expert_requires_advantage = bool(loss_cfg.get("dual_teacher_expert_requires_anchor_advantage", True))
    advantage_margin = float(loss_cfg.get("dual_teacher_expert_advantage_margin", 0.0))
    protect_nonexpert = bool(loss_cfg.get("dual_teacher_anchor_protect_nonexpert", True))
    protect_expert_classes = bool(loss_cfg.get("dual_teacher_anchor_protect_expert_classes", False))

    expert_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("dual_teacher_expert_classes", loss_cfg.get("focus_ce_classes", []))
    }
    protected_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("dual_teacher_protected_classes", [])
    }
    expert_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in expert_classes
    }
    protected_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in protected_classes
    }

    def _class_mask(class_indices: set[int]) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_idx in class_indices:
            mask |= labels.eq(int(class_idx))
        return mask

    def _teacher_stats(teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prob = F.softmax(teacher.float(), dim=1)
        top2 = prob.topk(k=min(2, prob.size(1)), dim=1)
        pred = top2.indices[:, 0]
        conf = top2.values[:, 0]
        if top2.values.size(1) > 1:
            gap = top2.values[:, 0] - top2.values[:, 1]
        else:
            gap = torch.ones_like(conf)
        true_logit = teacher.float().gather(1, labels.view(-1, 1)).squeeze(1)
        one_hot = F.one_hot(labels, num_classes=teacher.size(1)).bool()
        other_logit = teacher.float().masked_fill(one_hot, -torch.finfo(teacher.float().dtype).max).max(dim=1).values
        true_margin = true_logit - other_logit
        return pred, conf, gap, true_margin

    def _distill_values(teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_logits = student_logits.float()
        t_logits = teacher.float()
        kl_values = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)
        true_student = s_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        true_teacher = t_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        one_hot = F.one_hot(labels, num_classes=s_logits.size(1)).bool()
        student_other = s_logits.masked_fill(one_hot, -torch.finfo(s_logits.dtype).max).max(dim=1).values
        teacher_other = t_logits.masked_fill(one_hot, -torch.finfo(t_logits.dtype).max).max(dim=1).values
        margin_barrier = F.relu((true_teacher - teacher_other).detach() - margin_tolerance - (true_student - student_other))
        if bool(loss_cfg.get("dual_teacher_margin_squared", True)):
            margin_barrier = margin_barrier.pow(2)
        return kl_values, margin_barrier

    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        expert_mask = _class_mask(expert_idx) if expert_idx else torch.zeros_like(labels, dtype=torch.bool)
        anchor_mask = torch.zeros_like(labels, dtype=torch.bool)
        if protected_idx:
            anchor_mask |= _class_mask(protected_idx)
        if protect_nonexpert:
            anchor_mask |= ~expert_mask
        if not protect_expert_classes:
            anchor_mask &= ~expert_mask

        expert_count = 0.0
        anchor_count = 0.0
        expert_active_classes = 0
        anchor_active_classes = 0
        expert_terms: list[torch.Tensor] = []
        anchor_terms: list[torch.Tensor] = []
        expert_conf_mean = 0.0
        anchor_conf_mean = 0.0

        anchor_true_margin: torch.Tensor | None = None
        anchor_pred: torch.Tensor | None = None
        if anchor_logits is not None:
            anchor_pred, anchor_teacher_conf, anchor_gap, anchor_true_margin = _teacher_stats(anchor_logits)
            if require_teacher_correct:
                anchor_mask &= anchor_pred.eq(labels)
            if anchor_conf > 0.0:
                anchor_mask &= anchor_teacher_conf.ge(anchor_conf)
            if anchor_margin > 0.0:
                anchor_mask &= anchor_gap.ge(anchor_margin)
            anchor_conf_mean = float(anchor_teacher_conf.detach().mean().cpu())
            anchor_kl, anchor_barrier = _distill_values(anchor_logits)
            for class_idx in labels.detach().unique().tolist():
                class_mask = labels.eq(int(class_idx)) & anchor_mask
                if not bool(class_mask.any()):
                    continue
                class_terms = [anchor_kl[class_mask].mean()]
                if margin_weight > 0.0:
                    class_terms.append(float(margin_weight) * anchor_barrier[class_mask].mean())
                anchor_terms.append(torch.stack(class_terms).sum())
                anchor_count += float(class_mask.sum().detach().cpu())
                anchor_active_classes += 1

        if expert_logits is not None:
            expert_pred, expert_teacher_conf, expert_gap, expert_true_margin = _teacher_stats(expert_logits)
            if require_teacher_correct:
                expert_mask &= expert_pred.eq(labels)
            if expert_conf > 0.0:
                expert_mask &= expert_teacher_conf.ge(expert_conf)
            if expert_margin > 0.0:
                expert_mask &= expert_gap.ge(expert_margin)
            if expert_requires_advantage and anchor_true_margin is not None:
                expert_mask &= expert_true_margin.ge(anchor_true_margin + advantage_margin)
            if bool(loss_cfg.get("dual_teacher_expert_anchor_wrong_only", False)):
                if anchor_pred is None:
                    raise ValueError(
                        "dual_teacher_expert_anchor_wrong_only requires anchor logits"
                    )
                expert_mask &= anchor_pred.ne(labels)
            expert_conf_mean = float(expert_teacher_conf.detach().mean().cpu())
            expert_kl, expert_barrier = _distill_values(expert_logits)
            for class_idx in labels.detach().unique().tolist():
                class_mask = labels.eq(int(class_idx)) & expert_mask
                if not bool(class_mask.any()):
                    continue
                class_terms = [expert_kl[class_mask].mean()]
                if margin_weight > 0.0:
                    class_terms.append(float(margin_weight) * expert_barrier[class_mask].mean())
                expert_terms.append(torch.stack(class_terms).sum())
                expert_count += float(class_mask.sum().detach().cpu())
                expert_active_classes += 1

        terms: list[torch.Tensor] = []
        if expert_terms and expert_weight > 0.0:
            terms.append(float(expert_weight) * torch.stack(expert_terms).mean())
        if anchor_terms and anchor_weight > 0.0:
            terms.append(float(anchor_weight) * torch.stack(anchor_terms).mean())

    if not terms:
        return student_logits.new_zeros(()), {
            "loss_dual_teacher_noharm": 0.0,
            "dual_teacher_expert_count": float(expert_count),
            "dual_teacher_anchor_count": float(anchor_count),
            "dual_teacher_expert_active_classes": float(expert_active_classes),
            "dual_teacher_anchor_active_classes": float(anchor_active_classes),
            "dual_teacher_expert_conf_mean": expert_conf_mean,
            "dual_teacher_anchor_conf_mean": anchor_conf_mean,
        }
    loss = float(total_weight) * torch.stack(terms).sum().to(dtype=student_logits.dtype)
    return loss, {
        "loss_dual_teacher_noharm": float(loss.detach().cpu()),
        "dual_teacher_expert_count": float(expert_count),
        "dual_teacher_anchor_count": float(anchor_count),
        "dual_teacher_expert_active_classes": float(expert_active_classes),
        "dual_teacher_anchor_active_classes": float(anchor_active_classes),
        "dual_teacher_expert_conf_mean": expert_conf_mean,
        "dual_teacher_anchor_conf_mean": anchor_conf_mean,
    }


def classwise_pareto_groupdro_loss(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """RSCD classwise hard-group training with teacher-protected no-harm terms.

    This is the task-adapted combination of group DRO/JTT and LwF/GEM-style
    protection. The focus side gives more gradient to currently hard RSCD
    composite classes or anchor-teacher error groups. The protection side keeps
    teacher-correct, high-confidence classes from losing their distribution or
    true-vs-neighbor margin. Both sides are averaged by class first, so gains on
    one water/concrete boundary cannot hide regressions on another class.
    """

    focus_weight = float(loss_cfg.get("classwise_pareto_groupdro_weight", 0.0))
    protect_weight = float(loss_cfg.get("classwise_pareto_groupdro_protect_weight", 0.0))
    if focus_weight <= 0.0 and protect_weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_classwise_pareto_groupdro": 0.0,
            "classwise_pareto_focus_count": 0.0,
            "classwise_pareto_protect_count": 0.0,
        }

    logits = model_out["logits"]
    focus_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("classwise_pareto_groupdro_focus_classes", loss_cfg.get("focus_ce_classes", []))
    }
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    exempt_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("classwise_pareto_groupdro_exempt_classes", [])
    }
    exempt_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in exempt_classes
    }

    requested_pairs: set[frozenset[str]] = set()
    pair_specs = loss_cfg.get("classwise_pareto_groupdro_focus_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        s_logits = logits.float()
        t_logits = teacher_logits.float()
        per_ce = F.cross_entropy(s_logits, labels, reduction="none")
        teacher_ce = F.cross_entropy(t_logits, labels, reduction="none")
        teacher_prob = F.softmax(t_logits, dim=1)
        teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
        teacher_pred = teacher_top2.indices[:, 0]
        teacher_conf = teacher_top2.values[:, 0]
        if teacher_top2.values.size(1) > 1:
            teacher_gap = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
        else:
            teacher_gap = torch.ones_like(teacher_conf)
        teacher_error = teacher_pred.ne(labels)

        temperature = max(float(loss_cfg.get("classwise_pareto_groupdro_temperature", 2.0)), 1e-3)
        kl_values = F.kl_div(
            F.log_softmax(s_logits / temperature, dim=1),
            F.softmax(t_logits / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)

        one_hot = F.one_hot(labels, num_classes=s_logits.size(1)).bool()
        neg_inf = -torch.finfo(s_logits.dtype).max
        student_other = s_logits.masked_fill(one_hot, neg_inf).max(dim=1).values
        teacher_other = t_logits.masked_fill(one_hot, neg_inf).max(dim=1).values
        student_true_margin = s_logits.gather(1, labels.view(-1, 1)).squeeze(1) - student_other
        teacher_true_margin = t_logits.gather(1, labels.view(-1, 1)).squeeze(1) - teacher_other
        margin_tolerance = max(float(loss_cfg.get("classwise_pareto_groupdro_margin_tolerance", 0.02)), 0.0)
        margin_barrier = F.relu(teacher_true_margin.detach() - margin_tolerance - student_true_margin)
        if bool(loss_cfg.get("classwise_pareto_groupdro_margin_squared", True)):
            margin_barrier = margin_barrier.pow(2)

    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in focus_idx:
        focus_mask |= labels.eq(int(idx))
    if bool(loss_cfg.get("classwise_pareto_groupdro_include_teacher_errors", True)):
        focus_mask = focus_mask | teacher_error
    if requested_pairs:
        idx_to_name = {int(idx): canonical_class_label(name) for idx, name in idx_to_class.items()}
        pair_mask = torch.zeros_like(labels, dtype=torch.bool)
        for pair in spec.hard_pairs:
            left = int(pair.left)
            right = int(pair.right)
            names = frozenset((idx_to_name.get(left, str(left)), idx_to_name.get(right, str(right))))
            if names in requested_pairs:
                pair_mask |= labels.eq(left) | labels.eq(right)
        focus_mask = focus_mask & pair_mask
    if exempt_idx:
        exempt_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in exempt_idx:
            exempt_mask |= labels.eq(int(idx))
        focus_mask = focus_mask & ~exempt_mask

    sample_priority = logits.new_ones(labels.shape, dtype=torch.float32)
    physics_extra = max(float(loss_cfg.get("classwise_pareto_groupdro_physics_extra", 0.0)), 0.0)
    evidence = model_out.get("evidence_stats")
    if physics_extra > 0.0 and isinstance(evidence, torch.Tensor):
        stats = evidence.float().to(device=logits.device)
        grad_std = stats[:, 5].clamp(0.0, 1.0)
        lap_mean = stats[:, 6].clamp(0.0, 1.0)
        contrast_mean = stats[:, 7].clamp(0.0, 1.0)
        specular = stats[:, 8].clamp(0.0, 1.0)
        dark_water = stats[:, 9].clamp(0.0, 1.0)
        wet = stats[:, 10].clamp(0.0, 1.0)
        rough = stats[:, 11].clamp(0.0, 1.0)
        erasure = stats[:, 12].clamp(0.0, 1.0)
        wet_film = torch.clamp(0.45 * wet + 0.25 * dark_water + 0.15 * specular + 0.15 * erasure, 0.0, 1.0)
        hidden_rough = wet_film * torch.sigmoid((0.085 - rough) * 28.0)
        visible_rough = torch.clamp(0.35 * rough + 0.25 * grad_std + 0.22 * lap_mean + 0.18 * contrast_mean, 0.0, 1.0)
        granular = torch.clamp(0.35 * grad_std + 0.35 * lap_mean + 0.20 * contrast_mean + 0.10 * rough, 0.0, 1.0)
        factor_table = spec.class_to_factor.to(device=labels.device)
        factors = factor_table.index_select(0, labels)
        friction = factors[:, 0]
        material = factors[:, 1]
        roughness = factors[:, 2]
        wet_or_water = friction.eq(1) | friction.eq(2)
        concrete = material.eq(FACTOR_LABELS["material"].index("concrete"))
        loose = material.ge(FACTOR_LABELS["material"].index("mud"))
        has_roughness = roughness.ge(1)
        mechanism_score = torch.zeros_like(sample_priority)
        mechanism_score = torch.where(wet_or_water & concrete & has_roughness, hidden_rough, mechanism_score)
        mechanism_score = torch.where((friction.eq(0) & concrete & has_roughness), visible_rough, mechanism_score)
        mechanism_score = torch.where(loose, granular, mechanism_score)
        sample_priority = (sample_priority + physics_extra * mechanism_score).clamp(1.0, 1.0 + physics_extra)

    terms: list[torch.Tensor] = []
    focus_count = 0.0
    focus_group_count = 0
    focus_loss_sum = 0.0
    focus_softmax_temp = max(float(loss_cfg.get("classwise_pareto_groupdro_group_temperature", 5.0)), 0.0)
    group_losses: list[torch.Tensor] = []
    for class_idx in labels.detach().unique().tolist():
        class_idx = int(class_idx)
        class_mask = labels.eq(class_idx) & focus_mask
        if not bool(class_mask.any()):
            continue
        idx = class_mask.nonzero(as_tuple=False).flatten()
        weights = sample_priority.index_select(0, idx).to(device=per_ce.device, dtype=per_ce.dtype)
        group_loss = (per_ce.index_select(0, idx) * weights).sum() / weights.sum().clamp_min(1e-6)
        group_losses.append(group_loss)
        focus_count += float(idx.numel())
        focus_group_count += 1
        focus_loss_sum += float(group_loss.detach().cpu())
    if focus_weight > 0.0 and group_losses:
        grouped = torch.stack(group_losses)
        if focus_softmax_temp > 0.0 and grouped.numel() > 1:
            dro_weights = F.softmax(focus_softmax_temp * grouped.detach(), dim=0)
            focus_loss = (dro_weights * grouped).sum()
        else:
            focus_loss = grouped.mean()
        terms.append(float(focus_weight) * focus_loss)

    protect_conf = float(loss_cfg.get("classwise_pareto_groupdro_protect_confidence", 0.70))
    protect_margin = float(loss_cfg.get("classwise_pareto_groupdro_protect_teacher_margin", 0.12))
    protect_mask = teacher_pred.eq(labels)
    if protect_conf > 0.0:
        protect_mask = protect_mask & teacher_conf.ge(protect_conf)
    if protect_margin > 0.0:
        protect_mask = protect_mask & teacher_gap.ge(protect_margin)
    if bool(loss_cfg.get("classwise_pareto_groupdro_protect_nonfocus_only", True)):
        protect_mask = protect_mask & ~focus_mask
    if exempt_idx:
        exempt_mask = torch.zeros_like(labels, dtype=torch.bool)
        for idx in exempt_idx:
            exempt_mask |= labels.eq(int(idx))
        protect_mask = protect_mask & ~exempt_mask

    protect_terms: list[torch.Tensor] = []
    protect_count = 0.0
    protect_group_count = 0
    protect_kl_sum = 0.0
    protect_margin_sum = 0.0
    kl_weight = float(loss_cfg.get("classwise_pareto_groupdro_kl_weight", 1.0))
    margin_weight = float(loss_cfg.get("classwise_pareto_groupdro_margin_weight", 1.0))
    ce_barrier_weight = float(loss_cfg.get("classwise_pareto_groupdro_ce_barrier_weight", 0.0))
    ce_tolerance = max(float(loss_cfg.get("classwise_pareto_groupdro_ce_tolerance", 0.02)), 0.0)
    no_flip_weight = float(loss_cfg.get("classwise_pareto_groupdro_no_flip_weight", 0.0))
    for class_idx in labels.detach().unique().tolist():
        class_idx = int(class_idx)
        class_mask = labels.eq(class_idx) & protect_mask
        if not bool(class_mask.any()):
            continue
        class_parts: list[torch.Tensor] = []
        if kl_weight > 0.0:
            class_kl = kl_values[class_mask].mean()
            class_parts.append(float(kl_weight) * class_kl)
            protect_kl_sum += float(class_kl.detach().cpu())
        if margin_weight > 0.0:
            class_margin = margin_barrier[class_mask].mean()
            class_parts.append(float(margin_weight) * class_margin)
            protect_margin_sum += float(class_margin.detach().cpu())
        if ce_barrier_weight > 0.0:
            ce_excess = F.relu(per_ce[class_mask] - teacher_ce[class_mask].detach() - ce_tolerance)
            class_parts.append(float(ce_barrier_weight) * ce_excess.pow(2).mean())
        if no_flip_weight > 0.0:
            idx = class_mask.nonzero(as_tuple=False).flatten()
            class_parts.append(
                float(no_flip_weight)
                * F.cross_entropy(s_logits.index_select(0, idx), teacher_pred.index_select(0, idx))
            )
        if class_parts:
            protect_terms.append(torch.stack(class_parts).sum())
            protect_count += float(class_mask.sum().detach().cpu())
            protect_group_count += 1
    if protect_weight > 0.0 and protect_terms:
        protect_group_losses = torch.stack(protect_terms)
        protect_loss = protect_group_losses.mean()
        terms.append(float(protect_weight) * protect_loss)

    if not terms:
        return logits.new_zeros(()), {
            "loss_classwise_pareto_groupdro": 0.0,
            "classwise_pareto_focus_count": focus_count,
            "classwise_pareto_protect_count": protect_count,
            "classwise_pareto_teacher_error_rate": float(teacher_error.float().detach().mean().cpu()),
        }

    loss = torch.stack(terms).sum().to(dtype=logits.dtype)
    return loss, {
        "loss_classwise_pareto_groupdro": float(loss.detach().cpu()),
        "classwise_pareto_focus_count": focus_count,
        "classwise_pareto_focus_groups": float(focus_group_count),
        "classwise_pareto_focus_loss_mean": focus_loss_sum / max(focus_group_count, 1),
        "classwise_pareto_protect_count": protect_count,
        "classwise_pareto_protect_groups": float(protect_group_count),
        "classwise_pareto_protect_kl_mean": protect_kl_sum / max(protect_group_count, 1),
        "classwise_pareto_protect_margin_mean": protect_margin_sum / max(protect_group_count, 1),
        "classwise_pareto_teacher_error_rate": float(teacher_error.float().detach().mean().cpu()),
        "classwise_pareto_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
        "classwise_pareto_teacher_margin_mean": float(teacher_gap.detach().mean().cpu()),
        "classwise_pareto_priority_mean": float(sample_priority.detach().mean().cpu()),
    }


def _class_mask_from_names(
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    class_names: list[str] | tuple[str, ...] | set[str],
) -> torch.Tensor:
    target_names = {canonical_class_label(name) for name in class_names}
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if not target_names:
        return mask
    for idx, name in idx_to_class.items():
        if canonical_class_label(name) in target_names:
            mask |= labels.eq(int(idx))
    return mask


def rscd_focus_protect_objectives(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """Build RSCD-specific weak-class and protected-class objectives.

    The focus objective targets the RSCD composite classes where split concrete
    film/roughness mechanisms have shown possible gains. The protection
    objective is computed on non-focus samples, optionally restricted to anchor
    teacher-correct, high-confidence regions. It is used only as a gradient
    constraint by the surgery step below.
    """

    weight = float(loss_cfg.get("rscd_pcgrad_focus_weight", 0.0))
    if weight <= 0.0:
        return None, None, {"rscd_pcgrad_active": 0.0}
    focus_classes = loss_cfg.get("rscd_pcgrad_focus_classes", loss_cfg.get("focus_ce_classes", []))
    focus_mask = _class_mask_from_names(labels, idx_to_class, focus_classes)
    if bool(loss_cfg.get("rscd_pcgrad_protect_focus_teacher_correct", False)):
        protect_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        protect_mask = ~focus_mask
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        logits = student_logits.float()
        per_ce = F.cross_entropy(logits, labels, reduction="none")
        teacher_prob = None
        teacher_conf = None
        teacher_margin = None
        if teacher_logits is not None:
            teacher_prob = F.softmax(teacher_logits.float(), dim=1)
            teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
            teacher_pred = teacher_top2.indices[:, 0]
            teacher_conf = teacher_top2.values[:, 0]
            if teacher_top2.values.size(1) > 1:
                teacher_margin = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
            else:
                teacher_margin = torch.ones_like(teacher_conf)
            if bool(loss_cfg.get("rscd_pcgrad_protect_teacher_correct", True)):
                protect_mask = protect_mask & teacher_pred.eq(labels)
            if bool(loss_cfg.get("rscd_pcgrad_focus_teacher_errors_only", False)):
                focus_mask = focus_mask & teacher_pred.ne(labels)
            protect_conf = float(loss_cfg.get("rscd_pcgrad_protect_confidence", 0.0))
            protect_margin = float(loss_cfg.get("rscd_pcgrad_protect_margin", 0.0))
            if protect_conf > 0.0:
                protect_mask = protect_mask & teacher_conf.ge(protect_conf)
            if protect_margin > 0.0:
                protect_mask = protect_mask & teacher_margin.ge(protect_margin)
        if not bool(focus_mask.any()) or not bool(protect_mask.any()):
            return None, None, {
                "rscd_pcgrad_active": 0.0,
                "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
                "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
            }
        focus_loss = per_ce[focus_mask].mean()
        protect_loss = per_ce[protect_mask].mean()
        protect_kl_weight = float(loss_cfg.get("rscd_pcgrad_protect_kl_weight", 0.0))
        if teacher_logits is not None and teacher_prob is not None and protect_kl_weight > 0.0:
            temperature = max(float(loss_cfg.get("rscd_pcgrad_protect_temperature", 2.0)), 1e-3)
            protect_kl = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction="none",
            ).sum(dim=1) * (temperature * temperature)
            protect_loss = protect_loss + protect_kl_weight * protect_kl[protect_mask].mean()
    logs = {
        "rscd_pcgrad_active": 1.0,
        "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
        "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
        "loss_rscd_pcgrad_focus": float(focus_loss.detach().cpu()),
        "loss_rscd_pcgrad_protect": float(protect_loss.detach().cpu()),
    }
    if teacher_conf is not None and teacher_margin is not None:
        logs.update(
            {
                "rscd_pcgrad_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
                "rscd_pcgrad_teacher_margin_mean": float(teacher_margin.detach().mean().cpu()),
            }
        )
    return focus_loss, protect_loss, logs


def _indices_from_class_names(
    idx_to_class: dict[int, str],
    class_names: list[str] | tuple[str, ...] | set[str],
) -> set[int]:
    target_names = {canonical_class_label(name) for name in class_names}
    return {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in target_names
    }


def _group_mask_from_indices(labels: torch.Tensor, indices: set[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in indices:
        mask |= labels.eq(int(idx))
    return mask


def _build_factor_group_masks(
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    base_mask: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    """Build RSCD factor-aware protection groups for no-harm gradient surgery."""

    min_count = max(int(loss_cfg.get("rscd_pcgrad_group_min_count", 1)), 1)
    explicit_groups = loss_cfg.get("rscd_pcgrad_protect_groups", [])
    groups: list[tuple[str, torch.Tensor]] = []
    if explicit_groups:
        for group in explicit_groups:
            if isinstance(group, dict):
                name = str(group.get("name", f"group_{len(groups)}"))
                class_names = group.get("classes", [])
            else:
                name = str(group)
                class_names = [str(group)]
            indices = _indices_from_class_names(idx_to_class, class_names)
            if not indices:
                continue
            mask = _group_mask_from_indices(labels, indices) & base_mask
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((name, mask))
        return groups

    mode = str(loss_cfg.get("rscd_pcgrad_protect_group_mode", "factor")).lower()
    class_to_factor = spec.class_to_factor.to(device=labels.device)
    if mode in {"class", "classes", "per_class"}:
        for class_idx in labels[base_mask].detach().unique().tolist():
            class_idx = int(class_idx)
            name = canonical_class_label(idx_to_class.get(class_idx, str(class_idx)))
            mask = labels.eq(class_idx) & base_mask
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((f"class:{name}", mask))
        return groups

    if mode in {"coarse", "road_state"}:
        coarse_defs = {
            "dry_asphalt": ("dry_asphalt_smooth", "dry_asphalt_slight", "dry_asphalt_severe"),
            "dry_concrete": ("dry_concrete_smooth", "dry_concrete_slight", "dry_concrete_severe"),
            "wet_water_asphalt": (
                "wet_asphalt_smooth",
                "wet_asphalt_slight",
                "wet_asphalt_severe",
                "water_asphalt_smooth",
                "water_asphalt_slight",
                "water_asphalt_severe",
            ),
            "wet_water_concrete": (
                "wet_concrete_smooth",
                "wet_concrete_slight",
                "wet_concrete_severe",
                "water_concrete_smooth",
                "water_concrete_slight",
                "water_concrete_severe",
            ),
            "snow_ice": ("fresh_snow", "melted_snow", "ice"),
            "granular": ("dry_mud", "wet_mud", "water_mud", "dry_gravel", "wet_gravel", "water_gravel"),
        }
        for group_name, names in coarse_defs.items():
            indices = _indices_from_class_names(idx_to_class, names)
            mask = _group_mask_from_indices(labels, indices) & base_mask
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((f"coarse:{group_name}", mask))
        return groups

    factors = class_to_factor.index_select(0, labels)
    axes = loss_cfg.get("rscd_pcgrad_protect_factor_axes", FACTOR_AXES)
    for axis in axes:
        axis = str(axis)
        if axis not in FACTOR_AXES:
            continue
        axis_i = FACTOR_AXES.index(axis)
        valid = base_mask & factors[:, axis_i].ge(0)
        for value_idx in factors[valid, axis_i].detach().unique().tolist():
            value_idx = int(value_idx)
            if value_idx < 0:
                continue
            value_name = FACTOR_LABELS[axis][value_idx]
            mask = valid & factors[:, axis_i].eq(value_idx)
            if int(mask.sum().detach().cpu()) >= min_count:
                groups.append((f"{axis}:{value_name}", mask))
    return groups


def rscd_physics_focus_priority(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """PhysicsTexture-guided focus weights for RSCD no-harm gradient surgery.

    Generic PCGrad treats all focus samples equally. RSCD errors are more
    structured: wet/water concrete roughness is often hidden by film, asphalt
    water severity depends on dark/specular water evidence, and dry-concrete
    roughness depends on visible high-frequency texture. This detached priority
    only scales the focus objective; protected groups still constrain the final
    gradient direction.
    """

    extra = max(float(loss_cfg.get("rscd_pcgrad_focus_physics_extra", 0.0)), 0.0)
    evidence = model_out.get("evidence_stats")
    if extra <= 0.0 or not isinstance(evidence, torch.Tensor):
        return None, {
            "rscd_pcgrad_focus_physics_weight_mean": 1.0,
            "rscd_pcgrad_focus_physics_active_rate": 0.0,
        }

    stats = evidence.detach().float().to(device=labels.device)
    grad_mean = stats[:, 4].clamp(0.0, 1.0)
    grad_std = stats[:, 5].clamp(0.0, 1.0)
    lap_mean = stats[:, 6].clamp(0.0, 1.0)
    contrast_mean = stats[:, 7].clamp(0.0, 1.0)
    specular = stats[:, 8].clamp(0.0, 1.0)
    dark_water = stats[:, 9].clamp(0.0, 1.0)
    wet = stats[:, 10].clamp(0.0, 1.0)
    rough = stats[:, 11].clamp(0.0, 1.0)
    erasure = stats[:, 12].clamp(0.0, 1.0)
    snow_ice = torch.maximum(stats[:, 13], stats[:, 14]).clamp(0.0, 1.0)

    wet_film = torch.clamp(0.48 * wet + 0.24 * dark_water + 0.16 * specular + 0.12 * erasure, 0.0, 1.0)
    hidden_concrete_roughness = wet_film * torch.sigmoid((0.080 - rough) * 32.0) * torch.sigmoid(
        (0.34 - snow_ice) * 18.0
    )
    visible_concrete_roughness = torch.clamp(
        0.34 * rough + 0.25 * grad_std + 0.22 * lap_mean + 0.19 * contrast_mean,
        0.0,
        1.0,
    )
    asphalt_water_severity = torch.clamp(
        0.38 * dark_water + 0.24 * wet + 0.20 * specular + 0.18 * erasure,
        0.0,
        1.0,
    )
    granular_texture = torch.clamp(
        0.34 * grad_std + 0.30 * lap_mean + 0.22 * contrast_mean + 0.14 * grad_mean,
        0.0,
        1.0,
    )

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    wet_or_water = friction.eq(1) | friction.eq(2)
    concrete = material.eq(FACTOR_LABELS["material"].index("concrete"))
    asphalt = material.eq(FACTOR_LABELS["material"].index("asphalt"))
    loose = material.ge(FACTOR_LABELS["material"].index("mud"))
    rough_family = roughness.ge(1)

    priority = torch.zeros_like(wet_film)
    priority = torch.where(wet_or_water & concrete & rough_family, hidden_concrete_roughness, priority)
    priority = torch.where(friction.eq(0) & concrete & rough_family, visible_concrete_roughness, priority)
    priority = torch.where(wet_or_water & asphalt & rough_family, asphalt_water_severity, priority)
    priority = torch.where(loose, granular_texture, priority)

    weights = (1.0 + float(extra) * priority).clamp(
        1.0,
        max(float(loss_cfg.get("rscd_pcgrad_focus_physics_max_weight", 1.65)), 1.0),
    )
    logs = {
        "rscd_pcgrad_focus_physics_weight_mean": float(weights.detach().mean().cpu()),
        "rscd_pcgrad_focus_physics_active_rate": float(weights.detach().gt(1.001).float().mean().cpu()),
        "rscd_pcgrad_focus_hidden_concrete_mean": float(hidden_concrete_roughness.detach().mean().cpu()),
        "rscd_pcgrad_focus_asphalt_water_mean": float(asphalt_water_severity.detach().mean().cpu()),
    }
    return weights, logs


def family_mechanism_router_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise the early family router with RSCD mechanism families.

    Route 0 is a protection route. Routes 1-3 correspond to the three hard
    mechanisms that previously traded off against each other: visible dry
    concrete roughness, wet/water asphalt film severity, and wet/water concrete
    hidden roughness / smooth-bridge ambiguity.
    """

    weight = float(loss_cfg.get("family_mechanism_router_weight", 0.0))
    router_logits_obj = model_out.get("family_route_logits")
    logits_ref = model_out["logits"]
    if weight <= 0.0 or router_logits_obj is None:
        return logits_ref.new_zeros(()), {"loss_family_mechanism_router": 0.0, "family_router_active": 0.0}

    if isinstance(router_logits_obj, dict):
        router_values = [value for value in router_logits_obj.values() if isinstance(value, torch.Tensor)]
        if not router_values:
            return logits_ref.new_zeros(()), {"loss_family_mechanism_router": 0.0, "family_router_active": 0.0}
        router_logits = torch.stack([value.float() for value in router_values], dim=0).mean(dim=0)
    elif isinstance(router_logits_obj, torch.Tensor):
        router_logits = router_logits_obj.float()
    else:
        return logits_ref.new_zeros(()), {"loss_family_mechanism_router": 0.0, "family_router_active": 0.0}

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    dry = friction.eq(FACTOR_LABELS["friction"].index("dry"))
    wet_or_water = friction.eq(FACTOR_LABELS["friction"].index("wet")) | friction.eq(
        FACTOR_LABELS["friction"].index("water")
    )
    asphalt = material.eq(FACTOR_LABELS["material"].index("asphalt"))
    concrete = material.eq(FACTOR_LABELS["material"].index("concrete"))
    smooth = roughness.eq(FACTOR_LABELS["roughness"].index("smooth"))
    slight = roughness.eq(FACTOR_LABELS["roughness"].index("slight"))
    severe = roughness.eq(FACTOR_LABELS["roughness"].index("severe"))
    rough_visible = roughness.eq(FACTOR_LABELS["roughness"].index("slight")) | roughness.eq(
        FACTOR_LABELS["roughness"].index("severe")
    )
    paved_rough = roughness.ge(FACTOR_LABELS["roughness"].index("smooth"))
    split_concrete_routes = bool(loss_cfg.get("family_router_split_concrete_film_routes", False)) and int(
        router_logits.shape[1]
    ) >= 6

    target = torch.zeros_like(labels)
    dry_concrete_route = dry & concrete & rough_visible
    asphalt_film_route = wet_or_water & asphalt & rough_visible
    concrete_bridge_route = wet_or_water & concrete & paved_rough
    concrete_smooth_route = wet_or_water & concrete & smooth
    concrete_slight_route = wet_or_water & concrete & slight
    concrete_severe_route = wet_or_water & concrete & severe
    if bool(loss_cfg.get("family_router_dry_as_protect", False)):
        dry_concrete_route = torch.zeros_like(dry_concrete_route)
    if bool(loss_cfg.get("family_router_asphalt_as_protect", False)):
        asphalt_film_route = torch.zeros_like(asphalt_film_route)
    if bool(loss_cfg.get("family_router_concrete_as_protect", False)):
        concrete_bridge_route = torch.zeros_like(concrete_bridge_route)
        concrete_smooth_route = torch.zeros_like(concrete_smooth_route)
        concrete_slight_route = torch.zeros_like(concrete_slight_route)
        concrete_severe_route = torch.zeros_like(concrete_severe_route)
    target = torch.where(dry_concrete_route, torch.ones_like(target), target)
    target = torch.where(asphalt_film_route, torch.full_like(target, 2), target)
    if split_concrete_routes:
        target = torch.where(concrete_smooth_route, torch.full_like(target, 3), target)
        target = torch.where(concrete_slight_route, torch.full_like(target, 4), target)
        target = torch.where(concrete_severe_route, torch.full_like(target, 5), target)
    else:
        target = torch.where(concrete_bridge_route, torch.full_like(target, 3), target)

    route_weights = torch.full_like(router_logits[:, 0], float(loss_cfg.get("family_router_protect_sample_weight", 0.35)))
    route_weights = torch.where(target.gt(0), torch.ones_like(route_weights), route_weights)
    route_weights = torch.where(dry_concrete_route, route_weights * float(loss_cfg.get("family_router_dry_weight", 1.10)), route_weights)
    route_weights = torch.where(
        asphalt_film_route,
        route_weights * float(loss_cfg.get("family_router_asphalt_weight", 1.35)),
        route_weights,
    )
    route_weights = torch.where(
        concrete_smooth_route | concrete_slight_route | concrete_severe_route if split_concrete_routes else concrete_bridge_route,
        route_weights * float(loss_cfg.get("family_router_concrete_weight", 1.45)),
        route_weights,
    )
    if split_concrete_routes:
        route_weights = torch.where(
            concrete_slight_route,
            route_weights * float(loss_cfg.get("family_router_concrete_slight_weight", 1.0)),
            route_weights,
        )
        route_weights = torch.where(
            concrete_severe_route,
            route_weights * float(loss_cfg.get("family_router_concrete_severe_weight", 1.0)),
            route_weights,
        )

    per_sample = F.cross_entropy(router_logits, target, reduction="none")
    loss = (per_sample * route_weights).sum() / route_weights.sum().clamp_min(1e-6)

    probs_for_loss = F.softmax(router_logits, dim=1)
    probs = probs_for_loss.detach()
    route_entropy = -(
        probs_for_loss.clamp_min(1e-8) * probs_for_loss.clamp_min(1e-8).log()
    ).sum(dim=1) / math.log(float(router_logits.shape[1]))
    entropy_weight = float(loss_cfg.get("family_router_entropy_weight", 0.0))
    if entropy_weight > 0.0:
        entropy_mask = target.gt(0) if bool(loss_cfg.get("family_router_entropy_active_only", True)) else torch.ones_like(target, dtype=torch.bool)
        if bool(entropy_mask.any()):
            entropy_values = route_entropy[entropy_mask]
            entropy_sample_weights = route_weights[entropy_mask]
            entropy_loss = (entropy_values * entropy_sample_weights).sum() / entropy_sample_weights.sum().clamp_min(1e-6)
            loss = loss + entropy_weight * entropy_loss.to(device=loss.device, dtype=loss.dtype)
        else:
            entropy_loss = route_entropy.new_zeros(())
    else:
        entropy_loss = route_entropy.new_zeros(())

    margin_weight = float(loss_cfg.get("family_router_target_margin_weight", 0.0))
    margin_target = float(loss_cfg.get("family_router_target_margin", 0.28))
    if margin_weight > 0.0:
        one_hot = F.one_hot(target.clamp_min(0), num_classes=router_logits.shape[1]).bool()
        target_prob = probs_for_loss.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
        other_prob = probs_for_loss.masked_fill(one_hot, -1.0).amax(dim=1)
        margin_mask = target.gt(0) if bool(loss_cfg.get("family_router_margin_active_only", True)) else torch.ones_like(target, dtype=torch.bool)
        if bool(margin_mask.any()):
            margin_violation = F.relu(float(margin_target) - (target_prob - other_prob)).pow(2)
            margin_loss = (
                margin_violation[margin_mask] * route_weights[margin_mask]
            ).sum() / route_weights[margin_mask].sum().clamp_min(1e-6)
            loss = loss + margin_weight * margin_loss.to(device=loss.device, dtype=loss.dtype)
        else:
            margin_loss = route_entropy.new_zeros(())
    else:
        margin_loss = route_entropy.new_zeros(())

    pred = router_logits.detach().argmax(dim=1)
    logs = {
        "loss_family_mechanism_router": float(loss.detach().cpu()),
        "family_router_active": 1.0,
        "family_router_acc": float(pred.eq(target).float().mean().cpu()),
        "family_router_protect_count": float(target.eq(0).sum().detach().cpu()),
        "family_router_dry_count": float(target.eq(1).sum().detach().cpu()),
        "family_router_asphalt_count": float(target.eq(2).sum().detach().cpu()),
        "family_router_concrete_count": float(target.ge(3).sum().detach().cpu() if split_concrete_routes else target.eq(3).sum().detach().cpu()),
        "family_router_protect_prob_mean": float(probs[:, 0].mean().cpu()),
        "family_router_active_prob_mean": float(probs[:, 1:].sum(dim=1).mean().cpu()),
        "family_router_entropy_mean": float(route_entropy.detach().mean().cpu()),
        "family_router_entropy_loss": float(entropy_loss.detach().cpu()),
        "family_router_margin_loss": float(margin_loss.detach().cpu()),
    }
    if split_concrete_routes:
        logs.update(
            {
                "family_router_concrete_smooth_count": float(target.eq(3).sum().detach().cpu()),
                "family_router_concrete_slight_count": float(target.eq(4).sum().detach().cpu()),
                "family_router_concrete_severe_count": float(target.eq(5).sum().detach().cpu()),
                "family_router_concrete_smooth_prob_mean": float(probs[:, 3].mean().cpu()),
                "family_router_concrete_slight_prob_mean": float(probs[:, 4].mean().cpu()),
                "family_router_concrete_severe_prob_mean": float(probs[:, 5].mean().cpu()),
            }
        )
    leak_weight = float(loss_cfg.get("family_router_protect_leak_weight", 0.0))
    if leak_weight > 0.0 and bool(target.eq(0).any()):
        leak = probs_for_loss[:, 1:].sum(dim=1)
        protect_leak = leak[target.eq(0)]
        leak_loss = protect_leak.mean().to(device=router_logits.device, dtype=router_logits.dtype)
        loss = loss + float(leak_weight) * leak_loss
        logs["family_router_protect_leak"] = float(protect_leak.detach().mean().cpu())
    return float(weight) * loss.to(dtype=logits_ref.dtype), logs


def rscd_focus_grouped_protect_objectives(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor | None, list[tuple[str, torch.Tensor]], dict[str, float]]:
    """Focus-vs-many-protect objectives for RSCD factor-group no-harm updates.

    A single non-focus protection loss can hide local regressions: a gain on a
    wet-concrete boundary may hurt only dry-asphalt severe, while the averaged
    non-focus gradient still looks harmless. This RSCD-specific variant splits
    protected samples by class, coarse road state, or factor value and performs
    gradient conflict checks against each group separately.
    """

    weight = float(loss_cfg.get("rscd_pcgrad_focus_weight", 0.0))
    if weight <= 0.0:
        return None, [], {"rscd_pcgrad_grouped_active": 0.0}
    student_logits = model_out["logits"]
    focus_classes = loss_cfg.get("rscd_pcgrad_focus_classes", loss_cfg.get("focus_ce_classes", []))
    focus_mask = _class_mask_from_names(labels, idx_to_class, focus_classes)
    if bool(loss_cfg.get("rscd_pcgrad_protect_focus_teacher_correct", False)):
        protect_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        protect_mask = ~focus_mask
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        logits = student_logits.float()
        per_ce = F.cross_entropy(logits, labels, reduction="none")
        teacher_prob = None
        teacher_conf = None
        teacher_margin = None
        if teacher_logits is not None:
            teacher_prob = F.softmax(teacher_logits.float(), dim=1)
            teacher_top2 = teacher_prob.topk(k=min(2, teacher_prob.size(1)), dim=1)
            teacher_pred = teacher_top2.indices[:, 0]
            teacher_conf = teacher_top2.values[:, 0]
            if teacher_top2.values.size(1) > 1:
                teacher_margin = teacher_top2.values[:, 0] - teacher_top2.values[:, 1]
            else:
                teacher_margin = torch.ones_like(teacher_conf)
            if bool(loss_cfg.get("rscd_pcgrad_protect_teacher_correct", True)):
                protect_mask = protect_mask & teacher_pred.eq(labels)
            if bool(loss_cfg.get("rscd_pcgrad_focus_teacher_errors_only", False)):
                focus_mask = focus_mask & teacher_pred.ne(labels)
            protect_conf = float(loss_cfg.get("rscd_pcgrad_protect_confidence", 0.0))
            protect_margin = float(loss_cfg.get("rscd_pcgrad_protect_margin", 0.0))
            if protect_conf > 0.0:
                protect_mask = protect_mask & teacher_conf.ge(protect_conf)
            if protect_margin > 0.0:
                protect_mask = protect_mask & teacher_margin.ge(protect_margin)
        if not bool(focus_mask.any()) or not bool(protect_mask.any()):
            return None, [], {
                "rscd_pcgrad_grouped_active": 0.0,
                "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
                "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
            }
        physics_weights, physics_logs = rscd_physics_focus_priority(model_out, labels, spec, loss_cfg)
        if physics_weights is not None:
            selected_weights = physics_weights[focus_mask].to(device=per_ce.device, dtype=per_ce.dtype)
            focus_loss = (per_ce[focus_mask] * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)
        else:
            focus_loss = per_ce[focus_mask].mean()
        group_masks = _build_factor_group_masks(labels, idx_to_class, spec, protect_mask, loss_cfg)
        protect_losses: list[tuple[str, torch.Tensor]] = []
        protect_kl_weight = float(loss_cfg.get("rscd_pcgrad_protect_kl_weight", 0.0))
        temperature = max(float(loss_cfg.get("rscd_pcgrad_protect_temperature", 2.0)), 1e-3)
        protect_kl = None
        if teacher_logits is not None and teacher_prob is not None and protect_kl_weight > 0.0:
            protect_kl = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits.float() / temperature, dim=1),
                reduction="none",
            ).sum(dim=1) * (temperature * temperature)
        for group_name, group_mask in group_masks:
            if not bool(group_mask.any()):
                continue
            protect_loss = per_ce[group_mask].mean()
            if protect_kl is not None:
                protect_loss = protect_loss + protect_kl_weight * protect_kl[group_mask].mean()
            protect_losses.append((group_name, protect_loss))
        max_groups = int(loss_cfg.get("rscd_pcgrad_max_protect_groups", 0))
        if max_groups > 0 and len(protect_losses) > max_groups:
            mode = str(loss_cfg.get("rscd_pcgrad_protect_group_select", "hardest")).lower()
            if mode in {"hard", "hardest", "loss"}:
                protect_losses = sorted(
                    protect_losses,
                    key=lambda item: float(item[1].detach().cpu()),
                    reverse=True,
                )[:max_groups]
            else:
                protect_losses = protect_losses[:max_groups]
    logs = {
        "rscd_pcgrad_grouped_active": 1.0 if protect_losses else 0.0,
        "rscd_pcgrad_focus_count": float(focus_mask.sum().detach().cpu()),
        "rscd_pcgrad_protect_count": float(protect_mask.sum().detach().cpu()),
        "rscd_pcgrad_protect_group_count": float(len(protect_losses)),
        "loss_rscd_pcgrad_focus": float(focus_loss.detach().cpu()),
    }
    if "physics_logs" in locals():
        logs.update(physics_logs)
    if protect_losses:
        logs["loss_rscd_pcgrad_protect_mean"] = float(
            torch.stack([loss.detach() for _, loss in protect_losses]).mean().cpu()
        )
    if teacher_conf is not None and teacher_margin is not None:
        logs.update(
            {
                "rscd_pcgrad_teacher_conf_mean": float(teacher_conf.detach().mean().cpu()),
                "rscd_pcgrad_teacher_margin_mean": float(teacher_margin.detach().mean().cpu()),
            }
        )
    return focus_loss, protect_losses, logs


def rscd_focus_protect_gradient_surgery(
    params: list[torch.nn.Parameter],
    focus_loss: torch.Tensor,
    protect_loss: torch.Tensor,
    *,
    focus_weight: float,
    accum: int,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Project weak-class gradients away from protected-class conflicts.

    This is a task-adapted PCGrad/GEM-style step. Let g_f minimize hard RSCD
    focus classes and g_p minimize protected non-focus samples. If
    <g_f, g_p> < 0, applying -g_f would increase the protected loss to first
    order, so the conflicting component of g_f is removed before it is added as
    an extra update on top of the normal training loss.
    """

    focus_grads = torch.autograd.grad(focus_loss, params, retain_graph=True, allow_unused=True)
    protect_grads = torch.autograd.grad(protect_loss, params, retain_graph=True, allow_unused=True)
    dot = focus_loss.new_zeros(())
    focus_norm = focus_loss.new_zeros(())
    protect_norm = focus_loss.new_zeros(())
    for focus_grad, protect_grad in zip(focus_grads, protect_grads):
        if focus_grad is not None:
            focus_norm = focus_norm + focus_grad.detach().pow(2).sum()
        if protect_grad is not None:
            protect_norm = protect_norm + protect_grad.detach().pow(2).sum()
        if focus_grad is not None and protect_grad is not None:
            dot = dot + (focus_grad.detach() * protect_grad.detach()).sum()
    protect_norm = protect_norm.clamp_min(1e-12)
    conflict = bool((dot < 0).detach().cpu())
    coeff = dot / protect_norm if conflict else dot.new_zeros(())
    scale = float(focus_weight) / max(int(accum), 1)
    adjusted: list[torch.Tensor | None] = []
    with torch.no_grad():
        for focus_grad, protect_grad in zip(focus_grads, protect_grads):
            if focus_grad is None:
                adjusted.append(None)
                continue
            if conflict and protect_grad is not None:
                update = focus_grad - coeff.to(dtype=focus_grad.dtype, device=focus_grad.device) * protect_grad
            else:
                update = focus_grad
            adjusted.append((float(scale) * update).detach())
    return adjusted, {
        "rscd_pcgrad_conflict": 1.0 if conflict else 0.0,
        "rscd_pcgrad_dot": float(dot.detach().cpu()),
        "rscd_pcgrad_focus_grad_norm": float(torch.sqrt(focus_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_pcgrad_protect_grad_norm": float(torch.sqrt(protect_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_pcgrad_projection_coeff": float(coeff.detach().cpu()),
        "rscd_pcgrad_focus_weight": float(focus_weight),
    }


def rscd_focus_grouped_protect_gradient_surgery(
    params: list[torch.nn.Parameter],
    focus_loss: torch.Tensor,
    protect_losses: list[tuple[str, torch.Tensor]],
    *,
    focus_weight: float,
    accum: int,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Sequential PCGrad against multiple RSCD protection groups."""

    focus_grads = torch.autograd.grad(focus_loss, params, retain_graph=True, allow_unused=True)
    current: list[torch.Tensor | None] = [
        grad.detach().clone() if grad is not None else None for grad in focus_grads
    ]
    focus_norm = focus_loss.new_zeros(())
    for grad in current:
        if grad is not None:
            focus_norm = focus_norm + grad.pow(2).sum()

    conflicts = 0
    protect_count = 0
    dot_sum = 0.0
    min_dot = None
    max_projection = 0.0
    for _, protect_loss in protect_losses:
        protect_grads = torch.autograd.grad(protect_loss, params, retain_graph=True, allow_unused=True)
        dot = protect_loss.new_zeros(())
        protect_norm = protect_loss.new_zeros(())
        for focus_grad, protect_grad in zip(current, protect_grads):
            if protect_grad is not None:
                protect_norm = protect_norm + protect_grad.detach().pow(2).sum()
            if focus_grad is not None and protect_grad is not None:
                dot = dot + (focus_grad * protect_grad.detach()).sum()
        protect_norm = protect_norm.clamp_min(1e-12)
        dot_value = float(dot.detach().cpu())
        dot_sum += dot_value
        min_dot = dot_value if min_dot is None else min(float(min_dot), dot_value)
        protect_count += 1
        if bool((dot < 0).detach().cpu()):
            conflicts += 1
            coeff = dot / protect_norm
            max_projection = max(max_projection, abs(float(coeff.detach().cpu())))
            next_current: list[torch.Tensor | None] = []
            with torch.no_grad():
                for focus_grad, protect_grad in zip(current, protect_grads):
                    if focus_grad is None:
                        next_current.append(None)
                    elif protect_grad is None:
                        next_current.append(focus_grad)
                    else:
                        next_current.append(
                            focus_grad - coeff.to(dtype=focus_grad.dtype, device=focus_grad.device) * protect_grad.detach()
                        )
            current = next_current

    scale = float(focus_weight) / max(int(accum), 1)
    adjusted = [
        None if grad is None else (float(scale) * grad).detach()
        for grad in current
    ]
    return adjusted, {
        "rscd_pcgrad_grouped_conflicts": float(conflicts),
        "rscd_pcgrad_grouped_protect_losses": float(protect_count),
        "rscd_pcgrad_grouped_conflict_rate": float(conflicts / max(protect_count, 1)),
        "rscd_pcgrad_grouped_dot_mean": float(dot_sum / max(protect_count, 1)),
        "rscd_pcgrad_grouped_dot_min": float(min_dot if min_dot is not None else 0.0),
        "rscd_pcgrad_grouped_focus_grad_norm": float(torch.sqrt(focus_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_pcgrad_grouped_max_projection_coeff": float(max_projection),
        "rscd_pcgrad_focus_weight": float(focus_weight),
    }


def rscd_collect_protect_memory_gradient(
    params: list[torch.nn.Parameter],
    protect_loss: torch.Tensor | None,
    protect_losses: list[tuple[str, torch.Tensor]] | None = None,
) -> list[torch.Tensor | None] | None:
    """Collect a detached protected-memory gradient for RSCD A-GEM projection."""

    objective: torch.Tensor | None = None
    if protect_losses:
        objective = torch.stack([loss for _, loss in protect_losses]).mean()
    elif protect_loss is not None:
        objective = protect_loss
    if objective is None:
        return None
    grads = torch.autograd.grad(objective, params, retain_graph=True, allow_unused=True)
    return [None if grad is None else grad.detach().clone() for grad in grads]


def rscd_project_total_gradient_against_memory(
    params: list[torch.nn.Parameter],
    memory_grads: list[torch.Tensor | None] | None,
) -> dict[str, float]:
    """Project the total update so protected-sample loss does not increase."""

    if not memory_grads:
        return {
            "rscd_agem_total_projection_active": 0.0,
            "rscd_agem_total_projection_conflict": 0.0,
        }
    ref_grad = next((param.grad for param in params if param.grad is not None), None)
    if ref_grad is None:
        return {
            "rscd_agem_total_projection_active": 0.0,
            "rscd_agem_total_projection_conflict": 0.0,
        }
    dot = ref_grad.new_zeros(())
    grad_norm = ref_grad.new_zeros(())
    memory_norm = ref_grad.new_zeros(())
    with torch.no_grad():
        for param, memory_grad in zip(params, memory_grads):
            if param.grad is not None:
                grad_norm = grad_norm + param.grad.detach().pow(2).sum()
            if memory_grad is not None:
                memory_norm = memory_norm + memory_grad.detach().pow(2).sum()
            if param.grad is not None and memory_grad is not None:
                mem = memory_grad.to(device=param.grad.device, dtype=param.grad.dtype)
                dot = dot + (param.grad.detach() * mem).sum()
        memory_norm = memory_norm.clamp_min(1e-12)
        conflict = bool((dot < 0).detach().cpu())
        coeff = dot / memory_norm if conflict else dot.new_zeros(())
        if conflict:
            for param, memory_grad in zip(params, memory_grads):
                if param.grad is None or memory_grad is None:
                    continue
                mem = memory_grad.to(device=param.grad.device, dtype=param.grad.dtype)
                param.grad.sub_(coeff.to(device=param.grad.device, dtype=param.grad.dtype) * mem)
    return {
        "rscd_agem_total_projection_active": 1.0,
        "rscd_agem_total_projection_conflict": 1.0 if conflict else 0.0,
        "rscd_agem_total_projection_dot": float(dot.detach().cpu()),
        "rscd_agem_total_projection_coeff": float(coeff.detach().cpu()),
        "rscd_agem_total_projection_grad_norm": float(torch.sqrt(grad_norm.detach().clamp_min(0.0)).cpu()),
        "rscd_agem_total_projection_memory_norm": float(torch.sqrt(memory_norm.detach().clamp_min(0.0)).cpu()),
    }


def balanced_softmax_training_logits(
    logits: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    """生成 Balanced Softmax 训练 logits；验证和推理绝不调用该校正。

    The default behavior is deliberately an exact no-op.  Balanced Softmax is
    enabled only when ``loss.balanced_softmax_class_counts`` is a non-empty
    mapping from RSCD class name to a positive training count.  For class ``c``
    it trains with

        adjusted_logit_c = raw_logit_c + log(training_count_c).

    直观解释：训练集长尾类别的先验不同，训练 CE 前对第 c 类加
    ``log(N_c)``；模型本身输出的 raw logits 不被覆盖。验证集仍直接使用 raw
    logits，避免把训练先验错误带入近均衡验证集。

    The raw model output is *not* modified, so validation and inference still
    use the ordinary 27-way logits.  Requiring all 27 counts here prevents a
    silent class-order mistake, which would make a long-tail experiment
    uninterpretable.
    """

    raw_counts = loss_cfg.get("balanced_softmax_class_counts", {})
    if raw_counts in (None, {}):
        return logits
    if not isinstance(raw_counts, Mapping):
        raise TypeError(
            "loss.balanced_softmax_class_counts must be a class-name mapping"
        )
    if logits.ndim != 2:
        raise ValueError(
            "Balanced Softmax expects [batch, classes] logits, got "
            f"{tuple(logits.shape)}"
        )

    canonical_counts: dict[str, float] = {}
    for raw_name, raw_count in raw_counts.items():
        name = canonical_class_label(str(raw_name))
        count = float(raw_count)
        if not math.isfinite(count) or count <= 0.0:
            raise ValueError(
                "Balanced Softmax class counts must be finite and positive; "
                f"got {raw_name!r}={raw_count!r}"
            )
        if name in canonical_counts:
            raise ValueError(
                "duplicate canonical class in Balanced Softmax counts: "
                f"{name!r}"
            )
        canonical_counts[name] = count

    ordered_counts: list[float] = []
    missing: list[str] = []
    for class_idx in range(int(logits.shape[1])):
        if class_idx not in idx_to_class:
            raise ValueError(
                "Balanced Softmax requires idx_to_class for every logit; "
                f"missing index {class_idx}"
            )
        class_name = canonical_class_label(idx_to_class[class_idx])
        if class_name not in canonical_counts:
            missing.append(class_name)
            continue
        ordered_counts.append(canonical_counts[class_name])
    if missing:
        raise ValueError(
            "Balanced Softmax counts are missing classes: " + ", ".join(missing)
        )
    if len(canonical_counts) != len(ordered_counts):
        used = {
            canonical_class_label(idx_to_class[class_idx])
            for class_idx in range(int(logits.shape[1]))
        }
        extras = sorted(set(canonical_counts).difference(used))
        raise ValueError(
            "Balanced Softmax counts contain classes outside the classifier: "
            + ", ".join(extras)
        )

    # ordered_counts 严格按 checkpoint 的 class index 排序，不能依赖 YAML 字典顺序。
    log_counts = torch.tensor(
        ordered_counts,
        device=logits.device,
        dtype=logits.dtype,
    ).log()
    return logits + log_counts.unsqueeze(0)


def focus_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    """正式主分类损失入口；当前配置等价于 Balanced Softmax CE。

    ``focus_ce_extra_weight=0``，因此不会对个别类别人工加权。函数名保留是为了
    兼容历史实验，当前真正起作用的是 ``balanced_softmax_training_logits``。
    """
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("loss.label_smoothing must be in [0, 1)")
    # 仅用于算训练 CE 的临时 logits；原始 logits 继续用于准确率、验证和保存。
    training_logits = balanced_softmax_training_logits(
        logits,
        idx_to_class,
        loss_cfg,
    )
    extra = float(loss_cfg.get("focus_ce_extra_weight", 0.0))
    focus_classes = {canonical_class_label(name) for name in loss_cfg.get("focus_ce_classes", [])}
    if extra <= 0.0 or not focus_classes:
        return F.cross_entropy(
            training_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    focus_idx = {
        int(idx)
        for idx, name in idx_to_class.items()
        if canonical_class_label(name) in focus_classes
    }
    if not focus_idx:
        return F.cross_entropy(
            training_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in focus_idx:
        focus_mask |= labels.eq(int(idx))
    per_sample = F.cross_entropy(
        training_logits,
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    weights = 1.0 + float(extra) * focus_mask.to(dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def classifier_proxy_compactness_loss(
    model: nn.Module,
    model_out: dict[str, Any],
    labels: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pull embeddings toward their normalized classifier-weight proxies.

    This is a generic, class-name-free compactness objective for short
    from-scratch runs.  It reuses the 27-way linear classifier weights as
    proxies, so no class-center table or second optimizer is required.
    Detaching the proxies by default makes the term update the representation
    while cross-entropy remains responsible for positioning the classifier.
    """

    logits = model_out["logits"]
    weight = float(loss_cfg.get("proxy_compactness_weight", 0.0))
    if weight <= 0.0:
        return logits.new_zeros(()), {
            "loss_proxy_compactness": 0.0,
            "loss_proxy_compactness_weighted": 0.0,
            "proxy_target_cosine": 0.0,
        }
    feature = model_out.get("feature")
    if not isinstance(feature, torch.Tensor):
        feature = model_out.get("features")
    head = getattr(model, "head", None)
    head_weight = getattr(head, "weight", None)
    if not isinstance(feature, torch.Tensor) or not isinstance(
        head_weight,
        torch.Tensor,
    ):
        raise RuntimeError(
            "proxy_compactness_weight requires model_out['feature'] and "
            "a linear model.head.weight"
        )
    if feature.ndim != 2 or head_weight.ndim != 2:
        raise RuntimeError(
            "proxy compactness expects 2D embeddings and classifier weights"
        )
    if feature.shape[1] != head_weight.shape[1]:
        raise RuntimeError(
            "proxy compactness feature/head dimensions differ: "
            f"{feature.shape[1]} vs {head_weight.shape[1]}"
        )
    if labels.numel() and int(labels.max().detach().cpu()) >= head_weight.shape[0]:
        raise RuntimeError("proxy compactness labels exceed classifier rows")

    normalized_feature = F.normalize(feature.float(), dim=1, eps=1.0e-6)
    proxies = F.normalize(head_weight.float(), dim=1, eps=1.0e-6)
    if bool(loss_cfg.get("proxy_compactness_detach_proxy", True)):
        proxies = proxies.detach()
    target_proxy = proxies.index_select(0, labels)
    cosine = (normalized_feature * target_proxy).sum(dim=1)
    raw_loss = (1.0 - cosine).mean()
    weighted = float(weight) * raw_loss.to(dtype=logits.dtype)
    return weighted, {
        "loss_proxy_compactness": float(raw_loss.detach().cpu()),
        "loss_proxy_compactness_weighted": float(weighted.detach().cpu()),
        "proxy_target_cosine": float(cosine.detach().mean().cpu()),
    }


def mechanism_feature_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
    model_out: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """CE with RSCD mechanism-aware weights derived from PhysicsTexture values.

    The weights are not a separate classifier. They emphasize known RSCD hard
    boundaries when the image evidence indicates that the relevant factor is
    visually ambiguous: hidden wet/water roughness, visible dry-concrete
    roughness, asphalt water-film brightness, and granular mud/gravel texture.
    """

    extra = float(loss_cfg.get("feature_mechanism_ce_extra_weight", 0.0))
    evidence = model_out.get("evidence_stats")
    if extra <= 0.0 or not isinstance(evidence, torch.Tensor):
        return focus_weighted_cross_entropy(logits, labels, idx_to_class, loss_cfg), {
            "feature_mechanism_ce_weight_mean": 1.0,
            "feature_mechanism_ce_active_rate": 0.0,
        }

    stats = evidence.float()
    jitter_std = float(loss_cfg.get("feature_mechanism_ce_jitter_std", 0.0))
    if logits.requires_grad and jitter_std > 0.0:
        stats = (stats + torch.randn_like(stats) * jitter_std).clamp(0.0, 1.0)
    gray_std = stats[:, 1].clamp(0.0, 1.0)
    sat_std = stats[:, 3].clamp(0.0, 1.0)
    grad_mean = stats[:, 4].clamp(0.0, 1.0)
    grad_std = stats[:, 5].clamp(0.0, 1.0)
    lap_mean = stats[:, 6].clamp(0.0, 1.0)
    contrast_mean = stats[:, 7].clamp(0.0, 1.0)
    specular = stats[:, 8].clamp(0.0, 1.0)
    dark_water = stats[:, 9].clamp(0.0, 1.0)
    wet = stats[:, 10].clamp(0.0, 1.0)
    rough = stats[:, 11].clamp(0.0, 1.0)
    erasure = stats[:, 12].clamp(0.0, 1.0)
    snow_ice = torch.maximum(stats[:, 13], stats[:, 14]).clamp(0.0, 1.0)

    wet_film = torch.clamp(0.50 * wet + 0.25 * dark_water + 0.15 * specular + 0.10 * erasure, 0.0, 1.0)
    hidden_roughness = wet_film * torch.sigmoid((0.075 - rough) * 35.0) * torch.sigmoid((0.32 - snow_ice) * 18.0)
    visible_roughness = torch.clamp(0.35 * rough + 0.25 * grad_std + 0.20 * lap_mean + 0.20 * contrast_mean, 0.0, 1.0)
    dry_rough_ambiguity = (4.0 * visible_roughness * (1.0 - visible_roughness)).clamp(0.0, 1.0)
    asphalt_water_film = torch.clamp(0.45 * dark_water + 0.25 * wet + 0.20 * gray_std + 0.10 * sat_std, 0.0, 1.0)
    granular_texture = torch.clamp(0.35 * grad_std + 0.35 * lap_mean + 0.20 * contrast_mean + 0.10 * grad_mean, 0.0, 1.0)

    class_to_idx = {canonical_class_label(name): int(idx) for idx, name in idx_to_class.items()}

    def class_mask(names: tuple[str, ...]) -> torch.Tensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for name in names:
            idx = class_to_idx.get(canonical_class_label(name))
            if idx is not None:
                mask |= labels.eq(int(idx))
        return mask

    wet_water_concrete = class_mask(
        (
            "water_concrete_slight",
            "water_concrete_severe",
            "water_concrete_smooth",
            "wet_concrete_slight",
            "wet_concrete_severe",
            "wet_concrete_smooth",
        )
    )
    dry_concrete = class_mask(("dry_concrete_slight", "dry_concrete_severe", "dry_concrete_smooth"))
    water_asphalt = class_mask(
        (
            "water_asphalt_slight",
            "water_asphalt_severe",
            "water_asphalt_smooth",
            "wet_asphalt_slight",
            "wet_asphalt_severe",
        )
    )
    granular = class_mask(("water_mud", "water_gravel", "dry_mud", "dry_gravel", "wet_mud", "wet_gravel"))

    weights = logits.new_ones(labels.shape, dtype=torch.float32)
    weights = weights + float(extra) * hidden_roughness.to(device=labels.device) * wet_water_concrete.float()
    weights = weights + float(extra) * dry_rough_ambiguity.to(device=labels.device) * dry_concrete.float()
    weights = weights + float(extra) * asphalt_water_film.to(device=labels.device) * water_asphalt.float()
    weights = weights + float(extra) * granular_texture.to(device=labels.device) * granular.float()

    max_weight = max(float(loss_cfg.get("feature_mechanism_ce_max_weight", 1.8)), 1.0)
    weights = weights.clamp(1.0, max_weight).to(device=logits.device, dtype=logits.dtype)
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("loss.label_smoothing must be in [0, 1)")
    per_sample = F.cross_entropy(
        logits,
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    loss = (per_sample * weights).sum() / weights.sum().clamp_min(1.0)
    active = weights.detach().gt(1.001)
    logs = {
        "feature_mechanism_ce_weight_mean": float(weights.detach().mean().cpu()),
        "feature_mechanism_ce_active_rate": float(active.float().mean().cpu()),
        "feature_mechanism_ce_hidden_rough_mean": float(hidden_roughness.detach().mean().cpu()),
        "feature_mechanism_ce_visible_rough_mean": float(visible_roughness.detach().mean().cpu()),
    }
    return loss, logs


def anchor_error_gate_loss(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise learned pair gates with anchor pair-confusion targets."""

    weight = float(loss_cfg.get("anchor_error_gate_weight", 0.0))
    gate_logits = model_out.get("hardpair_error_gate_logits", {})
    if weight <= 0.0 or not isinstance(gate_logits, dict) or not gate_logits:
        return model_out["logits"].new_zeros(()), {"loss_anchor_error_gate": 0.0, "anchor_error_gate_count": 0.0}
    teacher_pred = teacher_logits.detach().argmax(dim=1)
    pos_weight_value = max(float(loss_cfg.get("anchor_error_gate_pos_weight", 8.0)), 1.0)
    pair_error_only = bool(loss_cfg.get("anchor_error_gate_pair_error_only", True))
    losses: list[torch.Tensor] = []
    gate_means: list[torch.Tensor] = []
    pos_count = 0
    total_count = 0
    for pair in spec.hard_pairs:
        key = f"p{int(pair.left)}_{int(pair.right)}"
        logit = gate_logits.get(key)
        if not isinstance(logit, torch.Tensor):
            continue
        left = int(pair.left)
        right = int(pair.right)
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        if pair_error_only:
            target = (mask_left & teacher_pred.eq(right)) | (mask_right & teacher_pred.eq(left))
        else:
            target = (mask_left | mask_right) & teacher_pred.ne(labels)
        idx = mask.nonzero(as_tuple=False).flatten()
        target_slice = target.index_select(0, idx).float()
        logit_slice = logit.index_select(0, idx).float()
        pos_weight = torch.as_tensor(pos_weight_value, device=logit_slice.device, dtype=logit_slice.dtype)
        losses.append(F.binary_cross_entropy_with_logits(logit_slice, target_slice, pos_weight=pos_weight))
        gate_means.append(torch.sigmoid(logit_slice.detach()).mean())
        pos_count += int(target_slice.detach().sum().cpu())
        total_count += int(target_slice.numel())
    if not losses:
        return model_out["logits"].new_zeros(()), {"loss_anchor_error_gate": 0.0, "anchor_error_gate_count": 0.0}
    loss = torch.stack(losses).mean().to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_anchor_error_gate": float(loss.detach().cpu()),
        "anchor_error_gate_count": float(total_count),
        "anchor_error_gate_pos_count": float(pos_count),
        "anchor_error_gate_pos_rate": float(pos_count / max(total_count, 1)),
    }
    if gate_means:
        logs["anchor_error_gate_mean"] = float(torch.stack(gate_means).mean().cpu())
    return float(weight) * loss, logs


def s7_drel_rescue_gate_loss(
    model_out: dict[str, Any],
    anchor_logits: torch.Tensor | None,
    expert_logits: torch.Tensor | None,
    labels: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Teach the DREL write gate to activate only on anchor errors it can rescue."""

    weight = float(loss_cfg.get("s7_drel_rescue_gate_weight", 0.0))
    gate_logits = model_out.get("s7_drel_rescue_gate_logits")
    empty = {
        "loss_s7_drel_rescue_gate": 0.0,
        "s7_drel_rescue_gate_count": 0.0,
        "s7_drel_rescue_gate_positive_count": 0.0,
    }
    if (
        weight <= 0.0
        or anchor_logits is None
        or expert_logits is None
        or not isinstance(gate_logits, torch.Tensor)
    ):
        return model_out["logits"].new_zeros(()), empty
    if gate_logits.ndim != 2 or int(gate_logits.shape[0]) != int(labels.numel()):
        raise RuntimeError(
            "s7_drel_rescue_gate_logits must be BxS and align with labels"
        )
    anchor_pred = anchor_logits.detach().argmax(dim=1)
    expert_pred = expert_logits.detach().argmax(dim=1)
    target = anchor_pred.ne(labels) & expert_pred.eq(labels)
    pos_weight = max(
        float(loss_cfg.get("s7_drel_rescue_gate_pos_weight", 8.0)),
        1.0,
    )
    target_matrix = target.float().unsqueeze(1).expand_as(gate_logits)
    raw_loss = F.binary_cross_entropy_with_logits(
        gate_logits.float(),
        target_matrix,
        pos_weight=torch.as_tensor(
            pos_weight,
            device=gate_logits.device,
            dtype=torch.float32,
        ),
    )
    loss = float(weight) * raw_loss.to(dtype=model_out["logits"].dtype)
    return loss, {
        "loss_s7_drel_rescue_gate": float(raw_loss.detach().cpu()),
        "s7_drel_rescue_gate_count": float(target_matrix.numel()),
        "s7_drel_rescue_gate_positive_count": float(target_matrix.detach().sum().cpu()),
        "s7_drel_rescue_gate_positive_rate": float(target.float().mean().detach().cpu()),
        "s7_drel_rescue_gate_mean": float(torch.sigmoid(gate_logits.detach()).mean().cpu()),
    }


def hardpair_margin_directed_loss(
    model_out: dict[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Push true-vs-confused hard-pair logits with explicit signed margins."""

    weight = float(loss_cfg.get("hardpair_margin_loss_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {"loss_hardpair_margin": 0.0, "hardpair_margin_count": 0.0}
    logits = model_out["logits"].float()
    teacher_pred = teacher_logits.detach().argmax(dim=1)
    target_margin = float(loss_cfg.get("hardpair_margin_target", 0.75))
    pair_error_only = bool(loss_cfg.get("hardpair_margin_teacher_pair_only", True))
    direction_weight = float(loss_cfg.get("hardpair_margin_direction_weight", 0.0))
    keep_weight = float(loss_cfg.get("hardpair_margin_keep_weight", 0.0))
    margin_delta = model_out.get("hardpair_margin_delta", {})
    losses: list[torch.Tensor] = []
    direction_losses: list[torch.Tensor] = []
    keep_losses: list[torch.Tensor] = []
    pos_count = 0
    total_count = 0
    margin_sum = 0.0
    delta_sum = 0.0
    delta_count = 0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        if pair_error_only:
            focus_mask = (mask_left & teacher_pred.eq(right)) | (mask_right & teacher_pred.eq(left))
        else:
            focus_mask = pair_mask & teacher_pred.ne(labels)
        if bool(focus_mask.any()):
            sign = torch.where(labels.eq(left), 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
            signed_margin = sign * (logits[:, left] - logits[:, right])
            selected_margin = signed_margin.index_select(0, focus_mask.nonzero(as_tuple=False).flatten())
            losses.append(F.relu(float(target_margin) - selected_margin).pow(2).mean())
            pos_count += int(focus_mask.sum().detach().cpu())
            margin_sum += float(selected_margin.detach().sum().cpu())
            key = f"p{left}_{right}"
            delta = margin_delta.get(key) if isinstance(margin_delta, dict) else None
            if direction_weight > 0.0 and isinstance(delta, torch.Tensor):
                selected_sign = sign.index_select(0, focus_mask.nonzero(as_tuple=False).flatten())
                selected_delta = delta.float().index_select(0, focus_mask.nonzero(as_tuple=False).flatten())
                direction_losses.append(F.relu(0.02 - selected_sign * selected_delta).pow(2).mean())
                delta_sum += float(selected_delta.detach().abs().sum().cpu())
                delta_count += int(selected_delta.numel())
        if keep_weight > 0.0:
            key = f"p{left}_{right}"
            delta = margin_delta.get(key) if isinstance(margin_delta, dict) else None
            keep_mask = pair_mask & teacher_pred.eq(labels)
            if isinstance(delta, torch.Tensor) and bool(keep_mask.any()):
                keep_losses.append(delta.float().index_select(0, keep_mask.nonzero(as_tuple=False).flatten()).pow(2).mean())
        total_count += int(pair_mask.sum().detach().cpu())
    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_hardpair_margin": 0.0,
            "hardpair_margin_count": float(total_count),
            "hardpair_margin_pos_count": 0.0,
        }
    loss = torch.stack(losses).mean()
    if direction_losses:
        loss = loss + float(direction_weight) * torch.stack(direction_losses).mean()
    if keep_losses:
        loss = loss + float(keep_weight) * torch.stack(keep_losses).mean()
    loss = loss.to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_hardpair_margin": float(loss.detach().cpu()),
        "hardpair_margin_count": float(total_count),
        "hardpair_margin_pos_count": float(pos_count),
        "hardpair_margin_pos_rate": float(pos_count / max(total_count, 1)),
        "hardpair_margin_selected_mean": float(margin_sum / max(pos_count, 1)),
    }
    if delta_count > 0:
        logs["hardpair_margin_abs_delta_mean"] = float(delta_sum / max(delta_count, 1))
    return float(weight) * loss, logs


def dry_concrete_bidirectional_ordinal_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Balanced two-sided ordinal loss for dry-concrete roughness boundaries.

    The feature-value screens showed a one-sided failure: correcting
    `dry_concrete_slight` from `dry_concrete_severe` can also push true severe
    samples into slight. This loss treats each configured dry-concrete hard
    pair as a bidirectional comparator and averages the two class-side losses,
    so both roughness directions must keep a margin.
    """

    weight = float(loss_cfg.get("dry_concrete_bidirectional_ordinal_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_dry_concrete_bidirectional": 0.0,
            "dry_concrete_bidirectional_count": 0.0,
        }

    logits = model_out["logits"].float()
    margin = float(loss_cfg.get("dry_concrete_bidirectional_margin", 0.55))
    low_margin = float(loss_cfg.get("dry_concrete_bidirectional_low_margin_threshold", -1.0))
    delta_weight = float(loss_cfg.get("dry_concrete_bidirectional_delta_weight", 0.0))
    delta_margin = float(loss_cfg.get("dry_concrete_bidirectional_delta_margin", 0.015))
    pair_specs = loss_cfg.get(
        "dry_concrete_bidirectional_pairs",
        ["dry_concrete_severe|dry_concrete_slight"],
    )
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]

    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) != 2:
            continue
        requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
    if not requested_pairs:
        return model_out["logits"].new_zeros(()), {
            "loss_dry_concrete_bidirectional": 0.0,
            "dry_concrete_bidirectional_count": 0.0,
        }

    idx_to_class = {idx: name for name, idx in spec.class_to_idx.items()}
    deltas = model_out.get("hardpair_margin_delta", {})
    losses: list[torch.Tensor] = []
    delta_losses: list[torch.Tensor] = []
    total = 0
    correct = 0
    side_terms = 0
    signed_gap_sum = 0.0
    selected_pair_count = 0

    for pair in spec.hard_pairs:
        if pair.axis != "roughness":
            continue
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(idx_to_class[left])
        right_name = canonical_class_label(idx_to_class[right])
        if frozenset((left_name, right_name)) not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        sign = torch.where(mask_left, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
        signed_gap = sign * (logits[:, left] - logits[:, right])
        focus_mask = pair_mask
        if low_margin >= 0.0:
            focus_mask = focus_mask & signed_gap.detach().le(low_margin)
        if not bool(focus_mask.any()):
            continue
        side_losses: list[torch.Tensor] = []
        for side_mask in (mask_left & focus_mask, mask_right & focus_mask):
            if bool(side_mask.any()):
                side_gap = signed_gap.index_select(0, side_mask.nonzero(as_tuple=False).flatten())
                side_losses.append(F.softplus(float(margin) - side_gap).mean())
                signed_gap_sum += float(side_gap.detach().sum().cpu())
                side_terms += int(side_gap.numel())
        if side_losses:
            losses.append(torch.stack(side_losses).mean())
            selected_pair_count += 1
        if delta_weight > 0.0 and isinstance(deltas, dict):
            key = f"p{left}_{right}"
            delta = deltas.get(key)
            if isinstance(delta, torch.Tensor):
                idx = focus_mask.nonzero(as_tuple=False).flatten()
                delta_slice = delta.float().index_select(0, idx)
                sign_slice = sign.index_select(0, idx)
                delta_losses.append(F.softplus(float(delta_margin) - sign_slice * delta_slice).mean())
        pair_idx = pair_mask.nonzero(as_tuple=False).flatten()
        pred_left = (logits[:, left] - logits[:, right]).index_select(0, pair_idx).ge(0.0)
        true_left = mask_left.index_select(0, pair_idx)
        correct += int(pred_left.eq(true_left).sum().detach().cpu())
        total += int(pair_idx.numel())

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_dry_concrete_bidirectional": 0.0,
            "dry_concrete_bidirectional_count": float(total),
        }
    loss = torch.stack(losses).mean()
    if delta_losses:
        loss = loss + float(delta_weight) * torch.stack(delta_losses).mean()
    loss = loss.to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_dry_concrete_bidirectional": float(loss.detach().cpu()),
        "dry_concrete_bidirectional_count": float(total),
        "dry_concrete_bidirectional_focus_count": float(side_terms),
        "dry_concrete_bidirectional_pair_count": float(selected_pair_count),
        "dry_concrete_bidirectional_pair_acc": float(correct / max(total, 1)),
        "dry_concrete_bidirectional_signed_gap_mean": float(signed_gap_sum / max(side_terms, 1)),
    }
    return float(weight) * loss, logs


def hardpair_binary_tournament_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train active hard-pair heads as RSCD left/right binary comparators."""

    weight = float(loss_cfg.get("hardpair_binary_loss_weight", 0.0))
    raw_logits = model_out.get("hardpair_margin_raw", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {"loss_hardpair_binary": 0.0, "hardpair_binary_count": 0.0}
    losses: list[torch.Tensor] = []
    total = 0
    correct = 0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        losses.append(F.binary_cross_entropy_with_logits(raw_slice, target))
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
    if not losses:
        return model_out["logits"].new_zeros(()), {"loss_hardpair_binary": 0.0, "hardpair_binary_count": 0.0}
    loss = torch.stack(losses).mean().to(dtype=model_out["logits"].dtype)
    return float(weight) * loss, {
        "loss_hardpair_binary": float(loss.detach().cpu()),
        "hardpair_binary_count": float(total),
        "hardpair_binary_acc": float(correct / max(total, 1)),
    }


def hardpair_value_adapter_pairwise_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise value adapters as pair-specific left/right classifiers."""

    weight = float(loss_cfg.get("hardpair_value_pairwise_loss_weight", 0.0))
    raw_logits = model_out.get("hardpair_value_adapter_logits", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {
            "loss_hardpair_value_pairwise": 0.0,
            "hardpair_value_pairwise_count": 0.0,
        }

    idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
    pair_weights_cfg = loss_cfg.get("hardpair_value_pairwise_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)
    loss_pairs_cfg = loss_cfg.get("hardpair_value_pairwise_loss_pairs", None)
    loss_pairs: set[frozenset[str]] = set()
    for item in loss_pairs_cfg or []:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            loss_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    losses: list[torch.Tensor] = []
    weighted_terms: list[torch.Tensor] = []
    total_weight = 0.0
    total = 0
    correct = 0
    active_pairs = 0
    logit_abs_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        term = F.binary_cross_entropy_with_logits(raw_slice, target)
        left_name = idx_to_class.get(left, str(left))
        right_name = idx_to_class.get(right, str(right))
        pair_names = frozenset((left_name, right_name))
        if loss_pairs and pair_names not in loss_pairs:
            continue
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(term)
        weighted_terms.append(term * pair_weight)
        total_weight += pair_weight
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
        active_pairs += 1
        logit_abs_sum += float(raw_slice.detach().abs().sum().cpu())

    if not weighted_terms:
        return model_out["logits"].new_zeros(()), {
            "loss_hardpair_value_pairwise": 0.0,
            "hardpair_value_pairwise_count": 0.0,
        }
    loss = (torch.stack(weighted_terms).sum() / max(total_weight, 1e-6)).to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_hardpair_value_pairwise": float(torch.stack(losses).mean().detach().cpu()),
        "loss_hardpair_value_pairwise_weighted": float(loss.detach().cpu()),
        "hardpair_value_pairwise_count": float(total),
        "hardpair_value_pairwise_pair_count": float(active_pairs),
        "hardpair_value_pairwise_acc": float(correct / max(total, 1)),
        "hardpair_value_pairwise_abs_logit_mean": float(logit_abs_sum / max(total, 1)),
    }
    return float(weight) * loss, logs


def feature_value_boundary_pairwise_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise feature-value boundary correctors on diagnosed hard pairs.

    The corrector is intentionally pair-local: a positive raw logit means the
    sample should move toward the left class in `spec.hard_pairs`, while a
    negative raw logit means it should move toward the right class.
    """

    weight = float(loss_cfg.get("feature_value_boundary_pairwise_loss_weight", 0.0))
    raw_logits = model_out.get("feature_value_boundary_logits", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {
            "loss_feature_value_boundary_pairwise": 0.0,
            "feature_value_boundary_pairwise_count": 0.0,
        }

    idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
    pair_weights_cfg = loss_cfg.get("feature_value_boundary_pairwise_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)
    loss_pairs_cfg = loss_cfg.get("feature_value_boundary_pairwise_loss_pairs", None)
    loss_pairs: set[frozenset[str]] = set()
    for item in loss_pairs_cfg or []:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            loss_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    losses: list[torch.Tensor] = []
    weighted_terms: list[torch.Tensor] = []
    total_weight = 0.0
    total = 0
    correct = 0
    active_pairs = 0
    logit_abs_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        left_name = idx_to_class.get(left, str(left))
        right_name = idx_to_class.get(right, str(right))
        pair_names = frozenset((left_name, right_name))
        if loss_pairs and pair_names not in loss_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        term = F.binary_cross_entropy_with_logits(raw_slice, target)
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(term)
        weighted_terms.append(term * pair_weight)
        total_weight += pair_weight
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
        active_pairs += 1
        logit_abs_sum += float(raw_slice.detach().abs().sum().cpu())

    if not weighted_terms:
        return model_out["logits"].new_zeros(()), {
            "loss_feature_value_boundary_pairwise": 0.0,
            "feature_value_boundary_pairwise_count": 0.0,
        }
    loss = (torch.stack(weighted_terms).sum() / max(total_weight, 1e-6)).to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_feature_value_boundary_pairwise": float(torch.stack(losses).mean().detach().cpu()),
        "loss_feature_value_boundary_pairwise_weighted": float(loss.detach().cpu()),
        "feature_value_boundary_pairwise_count": float(total),
        "feature_value_boundary_pairwise_pair_count": float(active_pairs),
        "feature_value_boundary_pairwise_acc": float(correct / max(total, 1)),
        "feature_value_boundary_pairwise_abs_logit_mean": float(logit_abs_sum / max(total, 1)),
    }
    return float(weight) * loss, logs


def water_concrete_opponent_feature_pairwise_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise feature-space opponent axes before the RSCD decoder.

    This loss is intentionally paired with `WaterConcreteOpponentFeatureConditioner`.
    A positive raw logit means the sample should move toward the left class of a
    hard pair; a negative raw logit means it should move toward the right class.
    Unlike S96's final-logit corrector, the supervised signal shapes the fused
    feature that feeds both factor tokens and the calibrated class head.
    """

    weight = float(loss_cfg.get("water_concrete_opponent_pairwise_loss_weight", 0.0))
    raw_logits = model_out.get("water_concrete_opponent_feature_logits", {})
    if weight <= 0.0 or not isinstance(raw_logits, dict) or not raw_logits:
        return model_out["logits"].new_zeros(()), {
            "loss_water_concrete_opponent_pairwise": 0.0,
            "water_concrete_opponent_pairwise_count": 0.0,
        }

    idx_to_class = {int(idx): canonical_class_label(name) for name, idx in spec.class_to_idx.items()}
    pair_specs = loss_cfg.get("water_concrete_opponent_pairwise_loss_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))

    pair_weights_cfg = loss_cfg.get("water_concrete_opponent_pairwise_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(
                pair_weight
            )

    weighted_terms: list[torch.Tensor] = []
    raw_terms: list[torch.Tensor] = []
    total_weight = 0.0
    total = 0
    correct = 0
    active_pairs = 0
    logit_abs_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        key = f"p{left}_{right}"
        raw = raw_logits.get(key)
        if not isinstance(raw, torch.Tensor):
            continue
        pair_names = frozenset((idx_to_class.get(left, str(left)), idx_to_class.get(right, str(right))))
        if requested_pairs and pair_names not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        mask = mask_left | mask_right
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).flatten()
        raw_slice = raw.float().index_select(0, idx)
        target = mask_left.float().index_select(0, idx)
        term = F.binary_cross_entropy_with_logits(raw_slice, target)
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        weighted_terms.append(term * pair_weight)
        raw_terms.append(term)
        total_weight += pair_weight
        pred_left = raw_slice.ge(0.0)
        correct += int(pred_left.eq(target.bool()).sum().detach().cpu())
        total += int(target.numel())
        active_pairs += 1
        logit_abs_sum += float(raw_slice.detach().abs().sum().cpu())

    if not weighted_terms:
        return model_out["logits"].new_zeros(()), {
            "loss_water_concrete_opponent_pairwise": 0.0,
            "water_concrete_opponent_pairwise_count": 0.0,
        }
    loss = (torch.stack(weighted_terms).sum() / max(total_weight, 1e-6)).to(dtype=model_out["logits"].dtype)
    logs = {
        "loss_water_concrete_opponent_pairwise": float(torch.stack(raw_terms).mean().detach().cpu()),
        "loss_water_concrete_opponent_pairwise_weighted": float(loss.detach().cpu()),
        "water_concrete_opponent_pairwise_count": float(total),
        "water_concrete_opponent_pairwise_pair_count": float(active_pairs),
        "water_concrete_opponent_pairwise_acc": float(correct / max(total, 1)),
        "water_concrete_opponent_pairwise_abs_logit_mean": float(logit_abs_sum / max(total, 1)),
    }
    return float(weight) * loss, logs


def value_guided_roughness_order_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
    idx_to_class: dict[int, str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Guide roughness factor ordering using physically visible texture evidence.

    This is deliberately not a post-logit correction. It supervises the
    intermediate roughness factor score so token-level roughness conditioning
    learns the RSCD ordinal relation smooth < slight < severe. The per-sample
    weight is lowered when wet-film/texture-erasure evidence suggests that
    roughness is visually unreliable.
    """

    weight = float(loss_cfg.get("value_guided_roughness_order_weight", 0.0))
    factor_logits = (
        model_out.get("c3_factor_logits", {})
        if model_out.get("factor_logits_source") == "backbone"
        else model_out.get("factor_logits", {})
    )
    rough_logits = factor_logits.get("roughness") if isinstance(factor_logits, dict) else None
    if weight <= 0.0 or not isinstance(rough_logits, torch.Tensor):
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_roughness_order": 0.0,
            "value_guided_roughness_order_count": 0.0,
        }

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    roughness = factors[:, 2]
    valid = torch.isin(roughness, torch.as_tensor([1, 2, 3], device=labels.device, dtype=roughness.dtype))
    focus_classes = {canonical_class_label(name) for name in loss_cfg.get("value_guided_roughness_order_classes", [])}
    if focus_classes:
        focus_idx = {
            int(idx)
            for idx, name in idx_to_class.items()
            if canonical_class_label(name) in focus_classes
        }
        focus_mask = torch.zeros_like(valid)
        for idx in focus_idx:
            focus_mask |= labels.eq(int(idx))
        valid = valid & focus_mask
    focus_friction = {str(item) for item in loss_cfg.get("value_guided_roughness_order_friction", [])}
    if focus_friction:
        friction_labels = FACTOR_LABELS["friction"]
        allowed = {
            idx
            for idx, name in enumerate(friction_labels)
            if str(name) in focus_friction
        }
        friction = factors[:, 0]
        friction_mask = torch.zeros_like(valid)
        for idx in allowed:
            friction_mask |= friction.eq(int(idx))
        valid = valid & friction_mask
    focus_material = {str(item) for item in loss_cfg.get("value_guided_roughness_order_material", [])}
    if focus_material:
        material_labels = FACTOR_LABELS["material"]
        allowed = {
            idx
            for idx, name in enumerate(material_labels)
            if str(name) in focus_material
        }
        material = factors[:, 1]
        material_mask = torch.zeros_like(valid)
        for idx in allowed:
            material_mask |= material.eq(int(idx))
        valid = valid & material_mask
    if not bool(valid.any()):
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_roughness_order": 0.0,
            "value_guided_roughness_order_count": 0.0,
        }

    idx = valid.nonzero(as_tuple=False).flatten()
    logits = rough_logits.float().index_select(0, idx)
    target = roughness.index_select(0, idx)
    evidence = model_out.get("evidence_stats")
    if isinstance(evidence, torch.Tensor):
        ev = evidence.float().index_select(0, idx)
        rough = ev[:, 11].clamp(0.0, 1.0)
        wet = ev[:, 10].clamp(0.0, 1.0)
        dark_water = ev[:, 9].clamp(0.0, 1.0)
        specular = ev[:, 8].clamp(0.0, 1.0)
        erasure = ev[:, 12].clamp(0.0, 1.0)
        visible = torch.sigmoid((rough - 0.018) * 130.0)
        occlusion_guard = torch.sigmoid((0.58 - erasure - 0.35 * wet - 0.25 * dark_water - 0.20 * specular) * 7.0)
        phys_weight = (visible * occlusion_guard).clamp(0.0, 1.0)
    else:
        phys_weight = logits.new_ones((idx.numel(),))
    rho = model_out.get("rho_roughness")
    if isinstance(rho, torch.Tensor):
        rho_weight = rho.float().view(-1).index_select(0, idx).clamp(0.0, 1.0)
        phys_weight = torch.maximum(phys_weight, 0.35 * rho_weight)
    min_weight = float(loss_cfg.get("value_guided_roughness_order_min_weight", 0.12))
    phys_weight = (min_weight + (1.0 - min_weight) * phys_weight).clamp(min_weight, 1.0)

    margin = float(loss_cfg.get("value_guided_roughness_order_margin", 0.55))
    slight_margin_scale = float(loss_cfg.get("value_guided_roughness_order_slight_margin_scale", 0.65))
    smooth = logits[:, 1]
    slight = logits[:, 2]
    severe = logits[:, 3]
    sample_losses: list[torch.Tensor] = []
    sample_weights: list[torch.Tensor] = []

    mask = target.eq(1)
    if bool(mask.any()):
        local_margin = margin * phys_weight[mask]
        sample_losses.append(
            0.5
            * (
                F.softplus(local_margin - (smooth[mask] - slight[mask]))
                + F.softplus(local_margin - (smooth[mask] - severe[mask]))
            )
        )
        sample_weights.append(phys_weight[mask])

    mask = target.eq(2)
    if bool(mask.any()):
        local_margin = margin * slight_margin_scale * phys_weight[mask]
        sample_losses.append(
            0.5
            * (
                F.softplus(local_margin - (slight[mask] - smooth[mask]))
                + F.softplus(local_margin - (slight[mask] - severe[mask]))
            )
        )
        sample_weights.append(phys_weight[mask])

    mask = target.eq(3)
    if bool(mask.any()):
        local_margin = margin * phys_weight[mask]
        sample_losses.append(
            0.5
            * (
                F.softplus(local_margin - (severe[mask] - slight[mask]))
                + F.softplus(local_margin - (severe[mask] - smooth[mask]))
            )
        )
        sample_weights.append(phys_weight[mask])

    if not sample_losses:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_roughness_order": 0.0,
            "value_guided_roughness_order_count": 0.0,
        }
    losses = torch.cat(sample_losses)
    weights = torch.cat(sample_weights).to(device=losses.device, dtype=losses.dtype)
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    pred = logits[:, 1:4].argmax(dim=1) + 1
    logs = {
        "loss_value_guided_roughness_order": float(loss.detach().cpu()),
        "value_guided_roughness_order_count": float(idx.numel()),
        "value_guided_roughness_order_weight_mean": float(phys_weight.detach().mean().cpu()),
        "value_guided_roughness_order_acc": float(pred.eq(target).float().mean().detach().cpu()),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def protected_tristate_roughness_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Ordinal class-logit loss for RSCD smooth/slight/severe triplets.

    The previous pair-value margin can help local boundaries, but it can also
    push severe concrete samples into the adjacent slight class. This loss
    treats roughness as a three-state ordered variable inside one fixed
    friction/material group and uses physical value evidence only as a
    reliability signal, not as a direct classifier.
    """

    weight = float(loss_cfg.get("protected_tristate_roughness_weight", 0.0))
    logits = model_out["logits"].float()
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_protected_tristate_roughness": 0.0,
            "protected_tristate_roughness_count": 0.0,
        }

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    valid = torch.isin(roughness, torch.as_tensor([1, 2, 3], device=labels.device, dtype=roughness.dtype))

    group_specs = loss_cfg.get("protected_tristate_roughness_groups", [])
    if isinstance(group_specs, str):
        group_specs = [group_specs]
    if group_specs:
        friction_labels = FACTOR_LABELS["friction"]
        material_labels = FACTOR_LABELS["material"]
        allowed: set[tuple[int, int]] = set()
        for item in group_specs:
            parts = str(item).replace("/", "|").replace(",", "|").split("|")
            if len(parts) != 2:
                continue
            f_name = canonical_class_label(parts[0])
            m_name = canonical_class_label(parts[1])
            if f_name in friction_labels and m_name in material_labels:
                allowed.add((friction_labels.index(f_name), material_labels.index(m_name)))
        if allowed:
            group_mask = torch.zeros_like(valid)
            for f_idx, m_idx in allowed:
                group_mask |= friction.eq(int(f_idx)) & material.eq(int(m_idx))
            valid = valid & group_mask

    grid = spec.class_index_grid.to(device=labels.device)
    smooth_idx = grid[friction.clamp_min(0), material.clamp_min(0), torch.ones_like(roughness)]
    slight_idx = grid[friction.clamp_min(0), material.clamp_min(0), torch.full_like(roughness, 2)]
    severe_idx = grid[friction.clamp_min(0), material.clamp_min(0), torch.full_like(roughness, 3)]
    valid = valid & smooth_idx.ge(0) & slight_idx.ge(0) & severe_idx.ge(0)
    if not bool(valid.any()):
        return model_out["logits"].new_zeros(()), {
            "loss_protected_tristate_roughness": 0.0,
            "protected_tristate_roughness_count": 0.0,
        }

    idx = valid.nonzero(as_tuple=False).flatten()
    group_indices = torch.stack(
        [
            smooth_idx.index_select(0, idx),
            slight_idx.index_select(0, idx),
            severe_idx.index_select(0, idx),
        ],
        dim=1,
    )
    group_logits = logits.gather(1, group_indices)
    target_rough = roughness.index_select(0, idx)
    target_rank = target_rough - 1

    values = model_out.get("hardpair_pair_value_evidence_vector")
    if isinstance(values, torch.Tensor):
        v = values.float().index_select(0, idx).clamp(0.0, 1.0)
        macro_rough = v[:, 0]
        micro_rough = v[:, 1]
        film = v[:, 2]
        artifact = v[:, 3]
        saturation = v[:, 4]
        macro_mean = v[:, 5]
        macro_std = v[:, 6]
        meso_std = v[:, 7]
        micro_std = v[:, 8]
        lap_std = v[:, 9]
        grad_std = v[:, 10]
        dark_water = v[:, 12]
        dark_water_top = v[:, 13]
        texture_erasure = v[:, 16]
        texture_erasure_top = v[:, 17]
        visible_rough = (
            0.30 * macro_rough
            + 0.18 * macro_std
            + 0.16 * meso_std
            + 0.12 * micro_rough
            + 0.10 * micro_std
            + 0.09 * lap_std
            + 0.05 * grad_std
        ).clamp(0.0, 1.0)
        film_occlusion = (
            0.34 * film
            + 0.24 * dark_water_top
            + 0.18 * dark_water
            + 0.14 * texture_erasure_top
            + 0.10 * texture_erasure
        ).clamp(0.0, 1.0)
        concrete_visibility = (0.55 * macro_mean + 0.25 * saturation + 0.20 * (1.0 - film)).clamp(0.0, 1.0)
        artifact_guard = (1.0 - 0.72 * artifact).clamp(0.10, 1.0)
    else:
        visible_rough = group_logits.new_full((idx.numel(),), 0.5)
        film_occlusion = group_logits.new_zeros((idx.numel(),))
        concrete_visibility = group_logits.new_full((idx.numel(),), 0.5)
        artifact_guard = group_logits.new_ones((idx.numel(),))

    selected_friction = friction.index_select(0, idx)
    wet_or_water = selected_friction.eq(1) | selected_friction.eq(2)
    dry = selected_friction.eq(0)
    wet_visibility_guard = torch.where(
        wet_or_water,
        (0.42 + 0.58 * (1.0 - film_occlusion)).clamp(0.12, 1.0),
        torch.ones_like(film_occlusion),
    )
    dry_visibility_boost = torch.where(dry, 1.0 + 0.20 * concrete_visibility, torch.ones_like(concrete_visibility))
    reliability = (artifact_guard * wet_visibility_guard * dry_visibility_boost).clamp(0.08, 1.0)
    rho = model_out.get("rho_roughness")
    if isinstance(rho, torch.Tensor):
        rho_weight = rho.float().view(-1).index_select(0, idx).clamp(0.0, 1.0)
        reliability = torch.maximum(reliability, 0.30 * rho_weight)

    min_weight = float(loss_cfg.get("protected_tristate_roughness_min_weight", 0.10))
    margin = float(loss_cfg.get("protected_tristate_roughness_margin", 0.48))
    severe_boost = float(loss_cfg.get("protected_tristate_roughness_severe_boost", 0.55))
    slight_protect = float(loss_cfg.get("protected_tristate_roughness_slight_protect", 0.55))
    smooth_score = ((1.0 - visible_rough) * (0.60 + 0.40 * film_occlusion)).clamp(0.0, 1.0)
    severe_score = (visible_rough * (1.0 - 0.45 * film_occlusion) * artifact_guard).clamp(0.0, 1.0)
    slight_score = (4.0 * visible_rough * (1.0 - visible_rough)).clamp(0.0, 1.0)

    smooth_logit = group_logits[:, 0]
    slight_logit = group_logits[:, 1]
    severe_logit = group_logits[:, 2]
    context_gate_enabled = bool(loss_cfg.get("protected_tristate_roughness_use_context_gate", False))
    if context_gate_enabled:
        with torch.no_grad():
            full_prob = F.softmax(logits, dim=1)
            group_mass = full_prob.gather(1, group_indices).sum(dim=1).clamp(0.0, 1.0)
            target_group_logit = group_logits.gather(1, target_rank.view(-1, 1)).squeeze(1)
            other_group_logits = group_logits.masked_fill(
                F.one_hot(target_rank, num_classes=3).bool(),
                torch.finfo(group_logits.dtype).min,
            )
            target_gap = target_group_logit - other_group_logits.max(dim=1).values
            mass_threshold = float(loss_cfg.get("protected_tristate_roughness_context_mass_threshold", 0.34))
            mass_temperature = float(loss_cfg.get("protected_tristate_roughness_context_mass_temperature", 12.0))
            gap_threshold = float(loss_cfg.get("protected_tristate_roughness_context_gap_threshold", 0.55))
            gap_temperature = float(loss_cfg.get("protected_tristate_roughness_context_gap_temperature", 4.0))
            floor = float(loss_cfg.get("protected_tristate_roughness_context_floor", 0.22))
            physics_context = (0.55 * concrete_visibility + 0.45 * (1.0 - film_occlusion)).clamp(0.0, 1.0)
            mass_gate = torch.sigmoid((group_mass - mass_threshold) * mass_temperature)
            hard_gate = torch.sigmoid((gap_threshold - target_gap) * gap_temperature)
            context_gate = (floor + (1.0 - floor) * mass_gate * hard_gate * physics_context).clamp(floor, 1.0)
    else:
        group_mass = group_logits.new_ones((idx.numel(),))
        target_gap = group_logits.gather(1, target_rank.view(-1, 1)).squeeze(1).detach()
        context_gate = group_logits.new_ones((idx.numel(),))
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    smooth_count = int(target_rough.eq(1).sum().detach().cpu())
    slight_count = int(target_rough.eq(2).sum().detach().cpu())
    severe_count = int(target_rough.eq(3).sum().detach().cpu())

    mask = target_rough.eq(1)
    if bool(mask.any()):
        local_weight = (
            min_weight + (1.0 - min_weight) * reliability[mask] * smooth_score[mask] * context_gate[mask]
        ).clamp(min_weight, 1.0)
        local_margin = margin * (0.55 + 0.45 * smooth_score[mask])
        losses.append(
            0.5
            * (
                F.softplus(local_margin - (smooth_logit[mask] - slight_logit[mask]))
                + F.softplus(0.65 * local_margin - (smooth_logit[mask] - severe_logit[mask]))
            )
        )
        weights.append(local_weight)

    mask = target_rough.eq(2)
    if bool(mask.any()):
        local_weight = (
            min_weight + (1.0 - min_weight) * reliability[mask] * slight_score[mask] * context_gate[mask]
        ).clamp(min_weight, 1.0)
        margin_vs_smooth = margin * (0.40 + 0.60 * slight_score[mask]) * (1.0 - 0.35 * smooth_score[mask])
        margin_vs_severe = margin * (0.40 + 0.60 * slight_score[mask]) * (1.0 - slight_protect * severe_score[mask])
        margin_vs_smooth = margin_vs_smooth.clamp_min(0.12 * margin)
        margin_vs_severe = margin_vs_severe.clamp_min(0.12 * margin)
        losses.append(
            0.5
            * (
                F.softplus(margin_vs_smooth - (slight_logit[mask] - smooth_logit[mask]))
                + F.softplus(margin_vs_severe - (slight_logit[mask] - severe_logit[mask]))
            )
        )
        weights.append(local_weight)

    mask = target_rough.eq(3)
    if bool(mask.any()):
        local_weight = (
            min_weight + (1.0 - min_weight) * reliability[mask] * severe_score[mask] * context_gate[mask]
        ).clamp(min_weight, 1.0)
        local_margin = margin * (0.60 + severe_boost * severe_score[mask])
        losses.append(
            0.5
            * (
                F.softplus(local_margin - (severe_logit[mask] - slight_logit[mask]))
                + F.softplus(0.75 * local_margin - (severe_logit[mask] - smooth_logit[mask]))
            )
        )
        weights.append(local_weight)

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_protected_tristate_roughness": 0.0,
            "protected_tristate_roughness_count": 0.0,
        }
    loss_values = torch.cat(losses)
    loss_weights = torch.cat(weights).to(device=loss_values.device, dtype=loss_values.dtype)
    loss = (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-6)
    pred_rank = group_logits.argmax(dim=1)
    severe_margin = (severe_logit - slight_logit).detach()
    slight_vs_severe = (slight_logit - severe_logit).detach()
    logs = {
        "loss_protected_tristate_roughness": float(loss.detach().cpu()),
        "protected_tristate_roughness_count": float(idx.numel()),
        "protected_tristate_roughness_acc": float(pred_rank.eq(target_rank).float().mean().detach().cpu()),
        "protected_tristate_roughness_weight_mean": float(loss_weights.detach().mean().cpu()),
        "protected_tristate_roughness_visible_mean": float(visible_rough.detach().mean().cpu()),
        "protected_tristate_roughness_occlusion_mean": float(film_occlusion.detach().mean().cpu()),
        "protected_tristate_roughness_smooth_count": float(smooth_count),
        "protected_tristate_roughness_slight_count": float(slight_count),
        "protected_tristate_roughness_severe_count": float(severe_count),
        "protected_tristate_roughness_severe_margin_mean": float(severe_margin.mean().cpu()),
        "protected_tristate_roughness_slight_vs_severe_mean": float(slight_vs_severe.mean().cpu()),
        "protected_tristate_roughness_context_gate_mean": float(context_gate.detach().mean().cpu()),
        "protected_tristate_roughness_group_mass_mean": float(group_mass.detach().mean().cpu()),
        "protected_tristate_roughness_target_gap_mean": float(target_gap.detach().mean().cpu()),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def roughness_fiber_conditional_ce_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise roughness directly inside the deployed 27-way logits.

    RSCD paved classes form complete three-state roughness fibers for each
    friction/material pair.  A separate auxiliary head can learn roughness
    without transferring that information to the deployed class decision.
    This loss instead gathers the smooth/slight/severe logits from the *same*
    friction/material fiber as the target and applies a conditional 3-way CE.

    The objective is parameter-free and does not move probability between
    friction or material states directly.  Undefined-roughness classes
    (snow, ice, mud and gravel) are masked.  With a zero weight it is exactly
    inactive, which makes weight-only matched ablations straightforward.
    """

    weight = float(loss_cfg.get("roughness_fiber_ce_weight", 0.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "loss.roughness_fiber_ce_weight must be finite and non-negative"
        )
    logits = model_out["logits"]
    if weight <= 0.0:
        return logits.sum() * 0.0, {
            "loss_roughness_fiber_ce": 0.0,
            "loss_roughness_fiber_ce_weighted": 0.0,
            "roughness_fiber_ce_count": 0.0,
            "roughness_fiber_ce_acc": 0.0,
        }

    factors = spec.class_to_factor.to(device=labels.device).index_select(0, labels)
    friction = factors[:, 0]
    material = factors[:, 1]
    roughness = factors[:, 2]
    defined = roughness.ge(1) & roughness.le(3)

    grid = spec.class_index_grid.to(device=labels.device)
    safe_friction = friction.clamp(min=0, max=int(grid.shape[0]) - 1)
    safe_material = material.clamp(min=0, max=int(grid.shape[1]) - 1)
    sibling_indices = torch.stack(
        [
            grid[safe_friction, safe_material, torch.full_like(roughness, state)]
            for state in (1, 2, 3)
        ],
        dim=1,
    )
    valid = defined & sibling_indices.ge(0).all(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0.0, {
            "loss_roughness_fiber_ce": 0.0,
            "loss_roughness_fiber_ce_weighted": 0.0,
            "roughness_fiber_ce_count": 0.0,
            "roughness_fiber_ce_acc": 0.0,
        }

    idx = valid.nonzero(as_tuple=False).flatten()
    selected_indices = sibling_indices.index_select(0, idx)
    group_logits = logits.float().index_select(0, idx).gather(1, selected_indices)
    target_rank = roughness.index_select(0, idx) - 1
    raw_loss = F.cross_entropy(group_logits, target_rank, reduction="mean")
    weighted_loss = float(weight) * raw_loss
    prediction = group_logits.argmax(dim=1)
    target_probability = F.softmax(group_logits.detach(), dim=1).gather(
        1, target_rank.view(-1, 1)
    )
    logs = {
        "loss_roughness_fiber_ce": float(raw_loss.detach().cpu()),
        "loss_roughness_fiber_ce_weighted": float(weighted_loss.detach().cpu()),
        "roughness_fiber_ce_count": float(idx.numel()),
        "roughness_fiber_ce_acc": float(
            prediction.eq(target_rank).float().mean().detach().cpu()
        ),
        "roughness_fiber_ce_target_probability_mean": float(
            target_probability.mean().cpu()
        ),
    }
    return weighted_loss.to(dtype=logits.dtype), logs


def value_guided_hardpair_margin_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train selected hard-pair margins with physics-weighted visibility.

    Unlike feature-value boundary correction, this loss does not add a new
    inference-time residual. It only tells the currently trainable middle
    mechanism to increase the true-vs-neighbor margin on selected RSCD hard
    pairs when physical roughness evidence is reliable enough.
    """

    weight = float(loss_cfg.get("value_guided_hardpair_margin_weight", 0.0))
    if weight <= 0.0:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_hardpair_margin": 0.0,
            "value_guided_hardpair_margin_count": 0.0,
        }
    pair_specs = loss_cfg.get("value_guided_hardpair_margin_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
    if not requested_pairs:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_hardpair_margin": 0.0,
            "value_guided_hardpair_margin_count": 0.0,
        }

    pair_weights_cfg = loss_cfg.get("value_guided_hardpair_margin_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)

    logits = model_out["logits"].float()
    idx_to_class = {idx: name for name, idx in spec.class_to_idx.items()}
    evidence = model_out.get("evidence_stats")
    rho = model_out.get("rho_roughness")
    margin = float(loss_cfg.get("value_guided_hardpair_margin_target", 0.48))
    low_margin = float(loss_cfg.get("value_guided_hardpair_margin_low_margin_threshold", 1.10))
    min_weight = float(loss_cfg.get("value_guided_hardpair_margin_min_weight", 0.10))
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    total = 0
    correct = 0
    selected_pairs = 0
    signed_margin_sum = 0.0
    physics_weight_sum = 0.0

    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(idx_to_class[left])
        right_name = canonical_class_label(idx_to_class[right])
        pair_names = frozenset((left_name, right_name))
        if pair_names not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        sign = torch.where(mask_left, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
        signed_margin = sign * (logits[:, left] - logits[:, right])
        focus_mask = pair_mask
        if low_margin >= 0.0:
            focus_mask = focus_mask & signed_margin.detach().le(low_margin)
        if not bool(focus_mask.any()):
            continue
        idx = focus_mask.nonzero(as_tuple=False).flatten()
        selected_margin = signed_margin.index_select(0, idx)
        phys_weight = selected_margin.new_ones(selected_margin.shape)
        if isinstance(evidence, torch.Tensor):
            ev = evidence.float().index_select(0, idx)
            rough = ev[:, 11].clamp(0.0, 1.0)
            wet = ev[:, 10].clamp(0.0, 1.0)
            dark_water = ev[:, 9].clamp(0.0, 1.0)
            specular = ev[:, 8].clamp(0.0, 1.0)
            erasure = ev[:, 12].clamp(0.0, 1.0)
            visible = torch.sigmoid((rough - 0.016) * 150.0)
            occlusion_guard = torch.sigmoid((0.60 - erasure - 0.30 * wet - 0.25 * dark_water - 0.18 * specular) * 7.5)
            phys_weight = (visible * occlusion_guard).to(device=selected_margin.device, dtype=selected_margin.dtype)
        if isinstance(rho, torch.Tensor):
            rho_weight = rho.float().view(-1).index_select(0, idx).to(dtype=selected_margin.dtype)
            phys_weight = torch.maximum(phys_weight, 0.30 * rho_weight.clamp(0.0, 1.0))
        phys_weight = (min_weight + (1.0 - min_weight) * phys_weight).clamp(min_weight, 1.0)
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(F.softplus(float(margin) - selected_margin))
        weights.append(phys_weight * pair_weight)
        pred_left = (logits[:, left] - logits[:, right]).index_select(0, idx).ge(0.0)
        true_left = mask_left.index_select(0, idx)
        correct += int(pred_left.eq(true_left).sum().detach().cpu())
        total += int(idx.numel())
        selected_pairs += 1
        signed_margin_sum += float(selected_margin.detach().sum().cpu())
        physics_weight_sum += float(phys_weight.detach().sum().cpu())

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_value_guided_hardpair_margin": 0.0,
            "value_guided_hardpair_margin_count": 0.0,
        }
    loss_values = torch.cat(losses)
    loss_weights = torch.cat(weights).to(device=loss_values.device, dtype=loss_values.dtype)
    loss = (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-6)
    logs = {
        "loss_value_guided_hardpair_margin": float(loss.detach().cpu()),
        "value_guided_hardpair_margin_count": float(total),
        "value_guided_hardpair_margin_pair_count": float(selected_pairs),
        "value_guided_hardpair_margin_acc": float(correct / max(total, 1)),
        "value_guided_hardpair_margin_signed_mean": float(signed_margin_sum / max(total, 1)),
        "value_guided_hardpair_margin_weight_mean": float(physics_weight_sum / max(total, 1)),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def pair_value_selective_margin_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use diagnosed pair-value evidence only as a training-time margin gate.

    This loss is the conservative follow-up to the value-augmentation audit:
    the physical/color/texture values are too weak as a classifier and can hurt
    if injected as an always-on residual. Here they only decide which samples
    are reliable enough to emphasize for selected hard-pair margins. Inference
    logits are unchanged.
    """

    weight = float(loss_cfg.get("pair_value_selective_margin_weight", 0.0))
    logits = model_out["logits"].float()
    value_vector = model_out.get("hardpair_pair_value_evidence_vector")
    if weight <= 0.0 or not isinstance(value_vector, torch.Tensor):
        return model_out["logits"].new_zeros(()), {
            "loss_pair_value_selective_margin": 0.0,
            "pair_value_selective_margin_count": 0.0,
        }

    pair_specs = loss_cfg.get("pair_value_selective_margin_pairs", [])
    if isinstance(pair_specs, str):
        pair_specs = [pair_specs]
    requested_pairs: set[frozenset[str]] = set()
    for item in pair_specs:
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            requested_pairs.add(frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1]))))
    if not requested_pairs:
        return model_out["logits"].new_zeros(()), {
            "loss_pair_value_selective_margin": 0.0,
            "pair_value_selective_margin_count": 0.0,
        }

    pair_weights_cfg = loss_cfg.get("pair_value_selective_margin_pair_weights", {}) or {}
    pair_weights: dict[frozenset[str], float] = {}
    for item, pair_weight in pair_weights_cfg.items():
        parts = str(item).replace("<->", "|").replace(",", "|").split("|")
        if len(parts) == 2:
            pair_weights[frozenset((canonical_class_label(parts[0]), canonical_class_label(parts[1])))] = float(pair_weight)

    value_aug_std = max(float(loss_cfg.get("pair_value_selective_margin_value_aug_std", 0.0)), 0.0)
    values = value_vector.float().clamp(0.0, 1.0)
    if value_aug_std > 0.0:
        values = (values + torch.randn_like(values) * value_aug_std).clamp(0.0, 1.0)

    idx_to_class = {idx: name for name, idx in spec.class_to_idx.items()}
    margin = float(loss_cfg.get("pair_value_selective_margin_target", 0.48))
    low_margin = float(loss_cfg.get("pair_value_selective_margin_low_margin_threshold", 1.05))
    min_weight = float(loss_cfg.get("pair_value_selective_margin_min_weight", 0.08))
    threshold = float(loss_cfg.get("pair_value_selective_margin_gate_threshold", 0.38))
    temperature = float(loss_cfg.get("pair_value_selective_margin_gate_temperature", 8.0))
    uncertainty_temperature = float(loss_cfg.get("pair_value_selective_margin_uncertainty_temperature", 2.0))

    def gate_for(pair_names: frozenset[str], local_values: torch.Tensor) -> torch.Tensor:
        macro_rough = local_values[:, 0].clamp(0.0, 1.0)
        micro_rough = local_values[:, 1].clamp(0.0, 1.0)
        film = local_values[:, 2].clamp(0.0, 1.0)
        artifact = local_values[:, 3].clamp(0.0, 1.0)
        saturation = local_values[:, 4].clamp(0.0, 1.0)
        macro_mean = local_values[:, 5].clamp(0.0, 1.0)
        macro_std = local_values[:, 6].clamp(0.0, 1.0)
        meso_std = local_values[:, 7].clamp(0.0, 1.0)
        micro_std = local_values[:, 8].clamp(0.0, 1.0)
        lap_std = local_values[:, 9].clamp(0.0, 1.0)
        grad_std = local_values[:, 10].clamp(0.0, 1.0)
        anisotropy = local_values[:, 11].clamp(0.0, 1.0)
        dark_water = local_values[:, 12].clamp(0.0, 1.0)
        dark_water_top = local_values[:, 13].clamp(0.0, 1.0)
        specular = local_values[:, 14].clamp(0.0, 1.0)
        specular_top = local_values[:, 15].clamp(0.0, 1.0)
        texture_erasure = local_values[:, 16].clamp(0.0, 1.0)
        texture_erasure_top = local_values[:, 17].clamp(0.0, 1.0)
        value_mean = local_values[:, 18].clamp(0.0, 1.0)
        value_std = local_values[:, 19].clamp(0.0, 1.0)
        if pair_names == frozenset(("dry_asphalt_slight", "dry_asphalt_severe")):
            score = 0.34 * macro_std + 0.24 * macro_mean + 0.18 * macro_rough + 0.14 * anisotropy + 0.10 * value_std
        elif pair_names == frozenset(("wet_asphalt_slight", "wet_asphalt_severe")):
            score = 0.36 * macro_std + 0.25 * macro_mean + 0.17 * macro_rough + 0.12 * anisotropy + 0.10 * saturation
        elif pair_names == frozenset(("water_asphalt_slight", "water_asphalt_severe")):
            score = 0.28 * macro_std + 0.22 * macro_mean + 0.18 * dark_water + 0.16 * meso_std + 0.16 * texture_erasure_top
        elif pair_names == frozenset(("dry_concrete_smooth", "dry_concrete_slight")):
            score = 0.28 * texture_erasure_top + 0.25 * meso_std + 0.19 * macro_mean + 0.16 * macro_std + 0.12 * value_std
        elif pair_names == frozenset(("dry_concrete_slight", "dry_concrete_severe")):
            score = 0.32 * macro_rough + 0.24 * macro_std + 0.18 * meso_std + 0.16 * lap_std + 0.10 * grad_std
        elif pair_names == frozenset(("water_asphalt_smooth", "water_asphalt_slight")):
            score = 0.31 * film + 0.24 * texture_erasure_top + 0.19 * micro_std + 0.16 * lap_std + 0.10 * dark_water_top
        elif pair_names == frozenset(("water_concrete_smooth", "water_concrete_slight")):
            score = 0.34 * dark_water_top + 0.26 * dark_water + 0.18 * film + 0.14 * texture_erasure + 0.08 * (1.0 - saturation)
        elif pair_names == frozenset(("wet_concrete_slight", "wet_concrete_severe")):
            score = 0.30 * anisotropy + 0.24 * macro_rough + 0.20 * texture_erasure_top + 0.16 * meso_std + 0.10 * saturation
        elif pair_names == frozenset(("water_gravel", "water_mud")):
            score = 0.30 * micro_rough + 0.23 * micro_std + 0.21 * lap_std + 0.16 * grad_std + 0.10 * saturation
        elif pair_names in {
            frozenset(("water_asphalt_slight", "wet_asphalt_slight")),
            frozenset(("water_asphalt_severe", "wet_asphalt_severe")),
            frozenset(("water_concrete_slight", "wet_concrete_slight")),
        }:
            score = 0.33 * dark_water_top + 0.25 * dark_water + 0.20 * film + 0.12 * specular_top + 0.10 * specular
        else:
            score = 0.25 * macro_rough + 0.25 * micro_rough + 0.20 * film + 0.15 * texture_erasure_top + 0.15 * lap_std
        artifact_guard = (1.0 - 0.70 * artifact).clamp(0.12, 1.0)
        gate = torch.sigmoid((score.clamp(0.0, 1.0) - threshold) * temperature) * artifact_guard
        return (min_weight + (1.0 - min_weight) * gate).clamp(min_weight, 1.0)

    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    total = 0
    correct = 0
    active_pairs = 0
    selected_margin_sum = 0.0
    gate_sum = 0.0
    for pair in spec.hard_pairs:
        left = int(pair.left)
        right = int(pair.right)
        left_name = canonical_class_label(idx_to_class[left])
        right_name = canonical_class_label(idx_to_class[right])
        pair_names = frozenset((left_name, right_name))
        if pair_names not in requested_pairs:
            continue
        mask_left = labels.eq(left)
        mask_right = labels.eq(right)
        pair_mask = mask_left | mask_right
        if not bool(pair_mask.any()):
            continue
        sign = torch.where(mask_left, 1.0, -1.0).to(device=logits.device, dtype=logits.dtype)
        signed_margin = sign * (logits[:, left] - logits[:, right])
        focus_mask = pair_mask
        if low_margin >= 0.0:
            focus_mask = focus_mask & signed_margin.detach().le(low_margin)
        if not bool(focus_mask.any()):
            continue
        idx = focus_mask.nonzero(as_tuple=False).flatten()
        selected_margin = signed_margin.index_select(0, idx)
        local_values = values.index_select(0, idx).to(device=selected_margin.device, dtype=selected_margin.dtype)
        sample_gate = gate_for(pair_names, local_values)
        if uncertainty_temperature > 0.0 and low_margin >= 0.0:
            uncertainty = torch.sigmoid((low_margin - selected_margin.detach()) * uncertainty_temperature)
            sample_gate = sample_gate * (0.35 + 0.65 * uncertainty.to(dtype=sample_gate.dtype))
        pair_weight = float(pair_weights.get(pair_names, 1.0))
        losses.append(F.softplus(float(margin) - selected_margin))
        weights.append(sample_gate * pair_weight)
        pred_left = (logits[:, left] - logits[:, right]).index_select(0, idx).ge(0.0)
        true_left = mask_left.index_select(0, idx)
        correct += int(pred_left.eq(true_left).sum().detach().cpu())
        total += int(idx.numel())
        active_pairs += 1
        selected_margin_sum += float(selected_margin.detach().sum().cpu())
        gate_sum += float(sample_gate.detach().sum().cpu())

    if not losses:
        return model_out["logits"].new_zeros(()), {
            "loss_pair_value_selective_margin": 0.0,
            "pair_value_selective_margin_count": 0.0,
        }
    loss_values = torch.cat(losses)
    loss_weights = torch.cat(weights).to(device=loss_values.device, dtype=loss_values.dtype)
    loss = (loss_values * loss_weights).sum() / loss_weights.sum().clamp_min(1e-6)
    logs = {
        "loss_pair_value_selective_margin": float(loss.detach().cpu()),
        "pair_value_selective_margin_count": float(total),
        "pair_value_selective_margin_pair_count": float(active_pairs),
        "pair_value_selective_margin_acc": float(correct / max(total, 1)),
        "pair_value_selective_margin_signed_mean": float(selected_margin_sum / max(total, 1)),
        "pair_value_selective_margin_gate_mean": float(gate_sum / max(total, 1)),
    }
    return float(weight) * loss.to(dtype=model_out["logits"].dtype), logs


def factor_marginal_consistency_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align 27-class probabilities with explicit factor probabilities.

    RSCD labels are compositional: every class is a friction/material/roughness
    triple. If the class posterior assigns mass to water-concrete classes, its
    friction and material marginals should agree with the factor heads. This
    loss ties the flat class distribution and the factorized tensor head without
    adding another classifier or post-hoc residual.
    """

    weight = float(loss_cfg.get("factor_marginal_consistency_weight", 0.0))
    factor_logits = (
        model_out.get("c3_factor_logits", {})
        if model_out.get("factor_logits_source") == "backbone"
        else model_out.get("factor_logits", {})
    )
    if weight <= 0.0 or not isinstance(factor_logits, dict):
        return model_out["logits"].new_zeros(()), {
            "loss_factor_marginal_consistency": 0.0,
            "factor_marginal_consistency_count": 0.0,
        }

    logits = model_out["logits"].float()
    temperature = max(float(loss_cfg.get("factor_marginal_consistency_temperature", 1.0)), 1.0e-3)
    class_probs = F.softmax(logits / temperature, dim=1)
    class_to_factor = spec.class_to_factor.to(device=logits.device)
    axis_weights_cfg = loss_cfg.get(
        "factor_marginal_consistency_axis_weights",
        {"friction": 1.0, "material": 1.0, "roughness": 1.0},
    )
    sample_weight = logits.new_ones((logits.shape[0],), dtype=torch.float32)
    downweight_classes = {
        canonical_class_label(name)
        for name in loss_cfg.get("factor_marginal_consistency_downweight_classes", [])
    }
    if downweight_classes:
        downweight_value = min(
            max(float(loss_cfg.get("factor_marginal_consistency_downweight_value", 0.25)), 0.0),
            1.0,
        )
        downweight_idx = {
            int(idx)
            for name, idx in spec.class_to_idx.items()
            if canonical_class_label(name) in downweight_classes
        }
        if downweight_idx:
            mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
            for idx in downweight_idx:
                mask |= labels.eq(int(idx))
            sample_weight = torch.where(
                mask.to(device=logits.device),
                sample_weight.new_full(sample_weight.shape, downweight_value),
                sample_weight,
            )
    roughness_axis_weight = sample_weight
    use_roughness_reliability_gate = bool(
        loss_cfg.get("factor_marginal_consistency_roughness_reliability_gate", False)
    )
    roughness_gate_logs: dict[str, float] = {}
    if use_roughness_reliability_gate:
        evidence = model_out.get("evidence_stats")
        gate_classes = {
            canonical_class_label(name)
            for name in loss_cfg.get("factor_marginal_consistency_roughness_gate_classes", [])
        }
        if isinstance(evidence, torch.Tensor) and gate_classes:
            gate_idx = {
                int(idx)
                for name, idx in spec.class_to_idx.items()
                if canonical_class_label(name) in gate_classes
            }
            gate_mask = torch.zeros_like(labels, dtype=torch.bool, device=labels.device)
            for idx in gate_idx:
                gate_mask |= labels.eq(int(idx))
            stats = evidence.to(device=logits.device, dtype=torch.float32)
            wet = stats[:, 10].clamp(0.0, 1.0)
            dark_water = stats[:, 9].clamp(0.0, 1.0)
            specular = stats[:, 8].clamp(0.0, 1.0)
            erasure = stats[:, 12].clamp(0.0, 1.0)
            wet_film = torch.clamp(
                0.45 * wet + 0.25 * dark_water + 0.15 * specular + 0.15 * erasure,
                0.0,
                1.0,
            )
            rho_target = C3PhysicsEvidenceStats.roughness_reliability_target(stats).view(-1).clamp(0.0, 1.0)
            film_occlusion = (wet_film * (1.0 - rho_target)).clamp(0.0, 1.0)
            strength = min(
                max(float(loss_cfg.get("factor_marginal_consistency_roughness_gate_strength", 0.85)), 0.0),
                1.0,
            )
            floor = min(
                max(float(loss_cfg.get("factor_marginal_consistency_roughness_gate_floor", 0.30)), 0.0),
                1.0,
            )
            reliability_gate = torch.clamp(1.0 - strength * film_occlusion, min=floor, max=1.0)
            roughness_axis_weight = torch.where(
                gate_mask.to(device=logits.device),
                sample_weight * reliability_gate.to(device=logits.device, dtype=sample_weight.dtype),
                sample_weight,
            )
            active = gate_mask.to(device=logits.device)
            if bool(active.any()):
                roughness_gate_logs = {
                    "factor_marginal_consistency_roughness_gate_active_rate": float(
                        active.float().mean().detach().cpu()
                    ),
                    "factor_marginal_consistency_roughness_gate_active_mean": float(
                        reliability_gate[active].detach().mean().cpu()
                    ),
                    "factor_marginal_consistency_roughness_gate_film_occlusion_active_mean": float(
                        film_occlusion[active].detach().mean().cpu()
                    ),
                    "factor_marginal_consistency_roughness_gate_rho_active_mean": float(
                        rho_target[active].detach().mean().cpu()
                    ),
                }
            else:
                roughness_gate_logs = {
                    "factor_marginal_consistency_roughness_gate_active_rate": 0.0,
                    "factor_marginal_consistency_roughness_gate_active_mean": 1.0,
                    "factor_marginal_consistency_roughness_gate_film_occlusion_active_mean": 0.0,
                    "factor_marginal_consistency_roughness_gate_rho_active_mean": 0.0,
                }
        else:
            use_roughness_reliability_gate = False
    eps = 1.0e-6
    terms: list[torch.Tensor] = []
    logs: dict[str, float] = {}

    for axis_idx, axis in enumerate(FACTOR_AXES):
        axis_logits = factor_logits.get(axis)
        if not isinstance(axis_logits, torch.Tensor):
            continue
        axis_logits = axis_logits.float() / temperature
        num_factor_classes = int(axis_logits.shape[1])
        factor_index = class_to_factor[:, axis_idx]
        valid = (factor_index >= 0) & (factor_index < num_factor_classes)
        if not bool(valid.any()):
            continue
        valid_index = factor_index[valid].long()
        valid_class_probs = class_probs[:, valid]
        marginal = logits.new_zeros((logits.shape[0], num_factor_classes))
        marginal.scatter_add_(
            1,
            valid_index.unsqueeze(0).expand(logits.shape[0], -1),
            valid_class_probs,
        )
        marginal = marginal.clamp_min(eps)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(eps)
        factor_prob = F.softmax(axis_logits, dim=1).clamp_min(eps)
        factor_prob = factor_prob / factor_prob.sum(dim=1, keepdim=True).clamp_min(eps)

        # Two one-way KL terms update both sides while using a stable detached target.
        factor_to_class = F.kl_div(factor_prob.log(), marginal.detach(), reduction="none").sum(dim=1)
        class_to_factor_term = F.kl_div(marginal.log(), factor_prob.detach(), reduction="none").sum(dim=1)
        axis_loss_values = 0.5 * (factor_to_class + class_to_factor_term)
        axis_sample_weight = roughness_axis_weight if axis == "roughness" else sample_weight
        axis_loss = (axis_loss_values * axis_sample_weight).sum() / axis_sample_weight.sum().clamp_min(eps)
        axis_weight = float(axis_weights_cfg.get(axis, 1.0)) if isinstance(axis_weights_cfg, dict) else 1.0
        terms.append(float(axis_weight) * axis_loss)
        logs[f"factor_marginal_consistency_{axis}"] = float(axis_loss.detach().cpu())
        logs[f"factor_marginal_consistency_{axis}_sample_weight_mean"] = float(
            axis_sample_weight.detach().mean().cpu()
        )
        logs[f"factor_marginal_consistency_{axis}_l1"] = float(
            (marginal.detach() - factor_prob.detach()).abs().mean().cpu()
        )

    if not terms:
        return model_out["logits"].new_zeros(()), {
            "loss_factor_marginal_consistency": 0.0,
            "factor_marginal_consistency_count": 0.0,
        }
    loss = torch.stack(terms).mean().to(dtype=model_out["logits"].dtype)
    logs["loss_factor_marginal_consistency"] = float(loss.detach().cpu())
    logs["factor_marginal_consistency_count"] = float(len(terms))
    logs["factor_marginal_consistency_sample_weight_mean"] = float(sample_weight.detach().mean().cpu())
    logs["factor_marginal_consistency_roughness_reliability_gate"] = float(use_roughness_reliability_gate)
    if use_roughness_reliability_gate:
        logs["factor_marginal_consistency_roughness_gate_mean"] = float(
            roughness_axis_weight.detach().mean().cpu()
        )
        logs.update(roughness_gate_logs)
    return float(weight) * loss, logs


def prepare_pareto_selected_edge_rules(loss_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load split-validation accepted hard-edge rules for train-time imitation."""

    rules: list[dict[str, Any]] = []
    path_value = loss_cfg.get("pareto_selected_edge_rules_path")
    if path_value:
        rules.extend(load_pareto_safe_logit_patch_rules(Path(str(path_value))))
    for item in loss_cfg.get("pareto_selected_edge_rules", []) or []:
        if not isinstance(item, dict):
            continue
        rule = item.get("rule_raw", item)
        if isinstance(rule, dict) and {"source", "target", "topk", "margin", "delta"}.issubset(rule):
            rules.append(
                {
                    "source": str(rule["source"]),
                    "target": str(rule["target"]),
                    "topk": int(rule["topk"]),
                    "margin": float(rule["margin"]),
                    "delta": float(rule["delta"]),
                }
            )
    seen: set[tuple[str, str, int, float, float]] = set()
    unique: list[dict[str, Any]] = []
    for rule in rules:
        key = (
            canonical_class_label(str(rule["source"])),
            canonical_class_label(str(rule["target"])),
            int(rule["topk"]),
            round(float(rule["margin"]), 6),
            round(float(rule["delta"]), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(rule)
    return unique


def pareto_selected_edge_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    rules: list[dict[str, Any]],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train only validation-accepted RSCD hard edges with source protection.

    A selected rule means that split validation allowed a local source->target
    boundary correction without Top-1/Macro-F1/protected-class regression. During
    training, the target class is pulled back only when it is in a close
    source-vs-target ambiguity, while true source samples receive a symmetric
    protection margin. The gates are detached so this remains a local boundary
    objective rather than a new global classifier.
    """

    weight = float(loss_cfg.get("pareto_selected_edge_loss_weight", 0.0))
    if weight <= 0.0 or not rules:
        return logits.new_zeros(()), {
            "loss_pareto_selected_edge": 0.0,
            "pareto_selected_edge_target_count": 0.0,
            "pareto_selected_edge_source_protect_count": 0.0,
        }
    class_to_idx = {canonical_class_label(name): int(idx) for idx, name in idx_to_class.items()}
    if not class_to_idx:
        return logits.new_zeros(()), {
            "loss_pareto_selected_edge": 0.0,
            "pareto_selected_edge_target_count": 0.0,
            "pareto_selected_edge_source_protect_count": 0.0,
        }

    target_margin = float(loss_cfg.get("pareto_selected_edge_target_margin", 0.10))
    source_margin = float(loss_cfg.get("pareto_selected_edge_source_margin", target_margin))
    source_weight = float(loss_cfg.get("pareto_selected_edge_source_protect_weight", 1.0))
    gate_temperature = float(loss_cfg.get("pareto_selected_edge_gate_temperature", 8.0))
    min_gate = float(loss_cfg.get("pareto_selected_edge_min_gate", 0.0))
    eps = 1e-6

    logits_f = logits.float()
    with torch.no_grad():
        probs = F.softmax(logits_f, dim=1)
        order = torch.argsort(logits_f, dim=1, descending=True)

    terms: list[torch.Tensor] = []
    target_count = 0.0
    source_count = 0.0
    active_rule_count = 0
    gate_sum = 0.0
    pair_correct = 0
    pair_total = 0
    for rule in rules:
        source_name = canonical_class_label(str(rule["source"]))
        target_name = canonical_class_label(str(rule["target"]))
        if source_name not in class_to_idx or target_name not in class_to_idx:
            continue
        source = int(class_to_idx[source_name])
        target = int(class_to_idx[target_name])
        topk = max(1, min(int(rule.get("topk", 2)), int(logits_f.shape[1])))
        gate_margin = float(rule.get("margin", loss_cfg.get("pareto_selected_edge_gate_margin", 0.35)))
        source_logit = logits_f[:, source]
        target_logit = logits_f[:, target]
        with torch.no_grad():
            pair_mass = (probs[:, source] + probs[:, target]).clamp(0.0, 1.0)
            abs_gap = (source_logit - target_logit).abs()
            boundary_gate = torch.sigmoid((gate_margin - abs_gap) * gate_temperature) * pair_mass
            if min_gate > 0.0:
                boundary_gate = min_gate + (1.0 - min_gate) * boundary_gate
            source_in_topk = order[:, :topk].eq(source).any(dim=1)
            target_in_topk = order[:, :topk].eq(target).any(dim=1)

        target_mask = labels.eq(target) & source_in_topk
        if bool(target_mask.any()):
            gate = boundary_gate[target_mask]
            margin = (target_logit - source_logit)[target_mask]
            loss_values = F.relu(target_margin - margin).pow(2)
            terms.append((loss_values * gate).sum() / gate.sum().clamp_min(eps))
            target_count += float(target_mask.sum().detach().cpu())
            gate_sum += float(gate.detach().sum().cpu())
            active_rule_count += 1

        source_mask = labels.eq(source) & target_in_topk
        if source_weight > 0.0 and bool(source_mask.any()):
            gate = boundary_gate[source_mask]
            margin = (source_logit - target_logit)[source_mask]
            loss_values = F.relu(source_margin - margin).pow(2)
            terms.append(float(source_weight) * (loss_values * gate).sum() / gate.sum().clamp_min(eps))
            source_count += float(source_mask.sum().detach().cpu())
            gate_sum += float(gate.detach().sum().cpu())
            active_rule_count += 1

        pair_mask = labels.eq(source) | labels.eq(target)
        if bool(pair_mask.any()):
            pred_target = target_logit[pair_mask].ge(source_logit[pair_mask])
            true_target = labels[pair_mask].eq(target)
            pair_correct += int(pred_target.eq(true_target).sum().detach().cpu())
            pair_total += int(pair_mask.sum().detach().cpu())

    if not terms:
        return logits.new_zeros(()), {
            "loss_pareto_selected_edge": 0.0,
            "pareto_selected_edge_target_count": float(target_count),
            "pareto_selected_edge_source_protect_count": float(source_count),
            "pareto_selected_edge_pair_acc": float(pair_correct / max(pair_total, 1)),
        }
    loss = torch.stack(terms).mean().to(dtype=logits.dtype)
    logs = {
        "loss_pareto_selected_edge": float(loss.detach().cpu()),
        "pareto_selected_edge_target_count": float(target_count),
        "pareto_selected_edge_source_protect_count": float(source_count),
        "pareto_selected_edge_active_rule_terms": float(active_rule_count),
        "pareto_selected_edge_gate_mean": float(gate_sum / max(target_count + source_count, 1.0)),
        "pareto_selected_edge_pair_acc": float(pair_correct / max(pair_total, 1)),
    }
    return float(weight) * loss, logs


def teacher_feature_distillation_loss(
    student_out: dict[str, Any],
    teacher_out: dict[str, Any] | None,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align a custom RSCD backbone to the frozen teacher's fused representation."""

    logits = student_out["logits"]
    weight = float(loss_cfg.get("teacher_feature_distill_weight", 0.0))
    if weight <= 0.0 or teacher_out is None:
        return logits.new_zeros(()), {
            "loss_teacher_feature_distill": 0.0,
            "teacher_feature_distill_cosine": 0.0,
            "teacher_feature_distill_count": 0.0,
        }
    key = str(loss_cfg.get("teacher_feature_distill_key", "feature"))
    student_feature = student_out.get(key)
    teacher_feature = teacher_out.get(key)
    if not isinstance(student_feature, torch.Tensor) or not isinstance(teacher_feature, torch.Tensor):
        return logits.new_zeros(()), {
            "loss_teacher_feature_distill": 0.0,
            "teacher_feature_distill_cosine": 0.0,
            "teacher_feature_distill_count": 0.0,
        }
    student_feature = student_feature.float()
    teacher_feature = teacher_feature.detach().float().to(device=student_feature.device)
    if student_feature.ndim > 2:
        student_feature = student_feature.flatten(1)
    if teacher_feature.ndim > 2:
        teacher_feature = teacher_feature.flatten(1)
    count = min(int(student_feature.shape[0]), int(teacher_feature.shape[0]))
    dim = min(int(student_feature.shape[1]), int(teacher_feature.shape[1]))
    if count <= 0 or dim <= 0:
        return logits.new_zeros(()), {
            "loss_teacher_feature_distill": 0.0,
            "teacher_feature_distill_cosine": 0.0,
            "teacher_feature_distill_count": 0.0,
        }
    student_feature = student_feature[:count, :dim]
    teacher_feature = teacher_feature[:count, :dim]
    student_norm = F.normalize(student_feature, dim=1)
    teacher_norm = F.normalize(teacher_feature, dim=1)
    cosine = (student_norm * teacher_norm).sum(dim=1).clamp(-1.0, 1.0)
    mode = str(loss_cfg.get("teacher_feature_distill_mode", "cosine")).lower()
    if mode == "mse":
        raw_loss = F.mse_loss(student_norm, teacher_norm)
    elif mode == "cosine_mse":
        raw_loss = (1.0 - cosine).mean() + 0.25 * F.mse_loss(student_norm, teacher_norm)
    else:
        raw_loss = (1.0 - cosine).mean()
    loss = float(weight) * raw_loss.to(dtype=logits.dtype)
    return loss, {
        "loss_teacher_feature_distill": float(raw_loss.detach().cpu()),
        "teacher_feature_distill_cosine": float(cosine.detach().mean().cpu()),
        "teacher_feature_distill_count": float(count),
    }


def backbone_stage_replacement_distillation_loss(
    model: nn.Module,
    logits: torch.Tensor,
    loss_cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Read a self-distillation objective produced by a shadow backbone stage.

    A progressive stage replacement can reuse the frozen parent stage as an
    online teacher without running a second full C3 model.  The backbone owns
    the feature-map comparison and exposes one scalar under ``last_aux``;
    this bridge merely validates and weights that scalar.  Keeping the loss
    generic avoids coupling the trainer to a particular replacement operator.
    """

    weight = float(loss_cfg.get("backbone_stage_distill_weight", 0.0) or 0.0)
    zero = logits.new_zeros(())
    empty_logs = {
        "loss_backbone_stage_distill": 0.0,
        "backbone_stage_distill_cosine": 0.0,
        "backbone_stage_distill_scale": 0.0,
        "backbone_stage_distill_count": 0.0,
    }
    if weight <= 0.0:
        return zero, empty_logs

    owner = getattr(model, "module", model)
    backbone = getattr(owner, "backbone", None)
    aux = getattr(backbone, "last_aux", None)
    if not isinstance(aux, Mapping):
        raise RuntimeError(
            "loss.backbone_stage_distill_weight requires "
            "model.backbone.last_aux to be a mapping"
        )
    raw = aux.get("stage_replacement_distill_loss")
    if not isinstance(raw, torch.Tensor) or raw.ndim != 0:
        raise RuntimeError(
            "shadow replacement backbone must expose scalar "
            "last_aux['stage_replacement_distill_loss']"
        )
    if not bool(torch.isfinite(raw.detach())):
        raise FloatingPointError("stage replacement distillation loss is non-finite")

    def _diagnostic(name: str) -> float:
        value = aux.get(name)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        detached = value.detach().float()
        if not bool(torch.isfinite(detached).all()):
            raise FloatingPointError(
                f"stage replacement diagnostic {name!r} is non-finite"
            )
        return float(detached.mean().cpu())

    count_value = aux.get("stage_replacement_distill_count")
    if isinstance(count_value, torch.Tensor):
        count = _diagnostic("stage_replacement_distill_count")
    else:
        count = float(count_value or 0.0)
    if not math.isfinite(count) or count <= 0.0:
        raise RuntimeError("stage replacement distillation count must be positive")

    raw_fp32 = raw.float()
    return weight * raw_fp32, {
        "loss_backbone_stage_distill": float(raw_fp32.detach().cpu()),
        "backbone_stage_distill_cosine": _diagnostic(
            "stage_replacement_distill_cosine"
        ),
        "backbone_stage_distill_scale": _diagnostic(
            "stage_replacement_distill_scale"
        ),
        "backbone_stage_distill_count": float(count),
    }


def _arcq_weak_gain_offset_view(
    image: torch.Tensor,
    *,
    gain_range: float,
    offset_range: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    field_grid_size: int,
) -> torch.Tensor:
    """Apply a low-frequency, label-preserving affine illumination field."""

    mean = mean.to(device=image.device, dtype=image.dtype)
    std = std.to(device=image.device, dtype=image.dtype)
    rgb = image * std + mean
    batch = int(image.shape[0])
    grid = max(int(field_grid_size), 1)
    gain = image.new_empty((batch, 1, grid, grid)).uniform_(
        -float(gain_range),
        float(gain_range),
    )
    offset = image.new_empty((batch, 1, grid, grid)).uniform_(
        -float(offset_range),
        float(offset_range),
    )
    if grid > 1:
        target_size = tuple(int(value) for value in image.shape[-2:])
        gain = F.interpolate(gain, size=target_size, mode="bilinear", align_corners=False)
        offset = F.interpolate(offset, size=target_size, mode="bilinear", align_corners=False)
    gain = 1.0 + gain
    weak_rgb = (rgb * gain + offset).clamp(0.0, 1.0)
    return (weak_rgb - mean) / std


def _arcq_direction_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Scale-free cosine consistency with an explicit zero-vector convention.

    Positive rescaling of either nonzero vector leaves the objective exactly
    unchanged.  Two zero vectors carry no contradictory directional evidence
    and therefore have zero loss; a zero/nonzero pair has unit loss.  The
    branch-safe denominator avoids the usual ``F.normalize(..., eps=...)``
    scale dependence for small but nonzero vectors.
    """

    if tuple(first.shape) != tuple(second.shape):
        raise ValueError(
            "ARCQ consistency tensors must have the same shape, got "
            f"{tuple(first.shape)} and {tuple(second.shape)}"
        )
    first_float = first.float()
    second_float = second.float()
    first_norm = torch.linalg.vector_norm(first_float, dim=1, keepdim=True)
    second_norm = torch.linalg.vector_norm(second_float, dim=1, keepdim=True)
    first_nonzero = first_norm > 0.0
    second_nonzero = second_norm > 0.0
    first_direction = first_float / torch.where(
        first_nonzero,
        first_norm,
        torch.ones_like(first_norm),
    )
    second_direction = second_float / torch.where(
        second_nonzero,
        second_norm,
        torch.ones_like(second_norm),
    )
    cosine = (first_direction * second_direction).sum(dim=1).clamp(-1.0, 1.0)
    both_zero = (~first_nonzero & ~second_nonzero).squeeze(1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return (1.0 - cosine).mean()


def arcq_composition_consistency_loss(
    model: nn.Module,
    image: torch.Tensor,
    model_out: dict[str, Any],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Empirically align protected-state direction under a weak light field.

    The spatial field and radiometric clipping are intentionally broader than
    the exact local shared-affine theorem. This is an augmentation regularizer,
    not a claim that the generated view remains inside the analytic model.
    """

    logits = model_out["logits"]
    weight = float(loss_cfg.get("arcq_composition_consistency_weight", 0.0))
    if weight <= 0.0:
        return logits.new_zeros(()), {"loss_arcq_composition_consistency": 0.0}
    sample_fraction = float(
        loss_cfg.get("arcq_composition_consistency_sample_fraction", 1.0)
    )
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(
            "arcq_composition_consistency_sample_fraction must be in (0, 1]"
        )
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise RuntimeError("ARCQ composition consistency requires model.backbone")
    if not bool(getattr(backbone, "use_composition_evidence", True)):
        raise ValueError(
            "ARCQ composition consistency requires "
            "use_composition_evidence=True; it is invalid for a "
            "composition-free control"
        )
    main_protected = model_out.get("arcq_protected_embedding")
    if not isinstance(main_protected, torch.Tensor):
        raise RuntimeError(
            "arcq_composition_consistency_weight requires arcq_protected_embedding in model output"
        )
    if bool(getattr(backbone, "allow_appearance_to_composition", False)):
        raise ValueError(
            "ARCQ composition consistency is undefined for the symmetric "
            "A-to-C control because forward_protected intentionally excludes "
            "appearance; disable consistency for that ablation"
        )
    mean = getattr(backbone, "input_mean", None)
    std = getattr(backbone, "input_std", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    batch_size = int(image.shape[0])
    sample_count = min(
        batch_size,
        max(1, int(math.ceil(sample_fraction * batch_size))),
    )
    if sample_count < batch_size:
        selected = torch.randperm(batch_size, device=image.device)[:sample_count]
        consistency_image = image.index_select(0, selected)
        main_protected = main_protected.index_select(0, selected)
    else:
        selected = None
        consistency_image = image
    weak_image = _arcq_weak_gain_offset_view(
        consistency_image,
        gain_range=float(loss_cfg.get("arcq_composition_gain_range", 0.08)),
        offset_range=float(loss_cfg.get("arcq_composition_offset_range", 0.03)),
        mean=mean,
        std=std,
        field_grid_size=int(loss_cfg.get("arcq_composition_field_grid_size", 4)),
    )
    protected_forward = getattr(backbone, "forward_protected", None)
    if not callable(protected_forward):
        raise RuntimeError(
            "ARCQ composition consistency requires backbone.forward_protected"
        )
    weak_state, weak_stages = protected_forward(
        weak_image,
        return_stage_embeddings=True,
    )
    if not isinstance(weak_state, torch.Tensor):
        raise RuntimeError("ARCQ protected-only forward did not return a tensor")
    weak_protected = weak_state.mean(dim=(2, 3))
    if tuple(weak_protected.shape) != tuple(main_protected.shape):
        raise RuntimeError(
            "ARCQ protected embedding shape changed between views: "
            f"{tuple(main_protected.shape)} vs {tuple(weak_protected.shape)}"
        )
    main_stages = model_out.get("arcq_protected_stage_embeddings")
    if (
        isinstance(main_stages, (tuple, list))
        and isinstance(weak_stages, (tuple, list))
        and len(main_stages) == len(weak_stages)
        and len(main_stages) > 0
    ):
        stage_losses = []
        for stage_index, (main_stage, weak_stage) in enumerate(
            zip(main_stages, weak_stages, strict=True)
        ):
            if not isinstance(main_stage, torch.Tensor) or not isinstance(
                weak_stage,
                torch.Tensor,
            ):
                raise RuntimeError(
                    f"ARCQ protected stage {stage_index} is not a tensor"
                )
            if selected is not None:
                main_stage = main_stage.index_select(0, selected)
            if tuple(main_stage.shape) != tuple(weak_stage.shape):
                raise RuntimeError(
                    "ARCQ protected stage shape changed between views: "
                    f"stage={stage_index} {tuple(main_stage.shape)} vs "
                    f"{tuple(weak_stage.shape)}"
                )
            stage_losses.append(
                _arcq_direction_consistency(weak_stage, main_stage)
            )
        raw_loss = torch.stack(stage_losses).sum()
    else:
        raw_loss = _arcq_direction_consistency(
            weak_protected,
            main_protected,
        )
    weighted = float(weight) * raw_loss.to(dtype=logits.dtype)
    main_rms = main_protected.float().square().mean().sqrt()
    weak_rms = weak_protected.float().square().mean().sqrt()
    main_variance = main_protected.float().var(dim=0, unbiased=False).mean()
    weak_variance = weak_protected.float().var(dim=0, unbiased=False).mean()
    return weighted, {
        "loss_arcq_composition_consistency": float(raw_loss.detach().cpu()),
        "loss_arcq_composition_consistency_weighted": float(weighted.detach().cpu()),
        "arcq_composition_consistency_samples": float(sample_count),
        "arcq_composition_consistency_realized_fraction": float(
            sample_count / max(batch_size, 1)
        ),
        "arcq_composition_main_rms": float(main_rms.detach().cpu()),
        "arcq_composition_weak_rms": float(weak_rms.detach().cpu()),
        "arcq_composition_main_batch_variance": float(
            main_variance.detach().cpu()
        ),
        "arcq_composition_weak_batch_variance": float(
            weak_variance.detach().cpu()
        ),
    }


def roughness_coral_aux_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the generic masked RSCD roughness CORAL auxiliary objective.

    ``roughness_coral_weight`` is the public key.  The historical
    ``arcq_roughness_coral_weight`` spelling remains a compatibility alias.
    A model may expose a fixed scalar ``roughness_aux_scale``; diagnostic
    controls use zero while still constructing and executing the same head.
    """

    logits = model_out["logits"]
    weight = float(
        loss_cfg.get(
            "roughness_coral_weight",
            loss_cfg.get("arcq_roughness_coral_weight", 0.0),
        )
    )
    if weight <= 0.0:
        return logits.new_zeros(()), {
            "loss_roughness_coral": 0.0,
            "loss_roughness_coral_weighted": 0.0,
            "roughness_aux_scale": 0.0,
        }
    roughness_logits = model_out.get("roughness_coral_logits")
    if not isinstance(roughness_logits, torch.Tensor):
        raise RuntimeError(
            "roughness_coral_weight (or its ARCQ compatibility alias) "
            "requires roughness_coral_logits in model output"
        )
    scale_value = model_out.get("roughness_aux_scale", 1.0)
    if isinstance(scale_value, torch.Tensor):
        if scale_value.numel() != 1:
            raise ValueError("roughness_aux_scale must be a scalar")
        scale = scale_value.to(device=logits.device, dtype=torch.float32).reshape(())
    else:
        scale = torch.tensor(
            float(scale_value),
            device=logits.device,
            dtype=torch.float32,
        )
    if not bool(torch.isfinite(scale.detach())):
        raise ValueError("roughness_aux_scale must be finite")
    raw_loss, logs = coral_roughness_loss(roughness_logits, labels, spec)
    weighted = (
        float(weight)
        * scale.to(dtype=raw_loss.dtype)
        * raw_loss
    ).to(dtype=logits.dtype)
    logs["loss_roughness_coral_weighted"] = float(weighted.detach().cpu())
    logs["roughness_aux_scale"] = float(scale.detach().cpu())
    return weighted, logs


def arcq_roughness_coral_aux_loss(
    model_out: dict[str, Any],
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Backward-compatible wrapper for older ARCQ callers and tests."""

    return roughness_coral_aux_loss(model_out, labels, spec, loss_cfg)


def _batch_valid_mask(
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    """Move an optional explicit support mask without fabricating one."""

    valid_mask = batch.get("valid_mask")
    if valid_mask is None:
        return None
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("batch.valid_mask must be a tensor")
    image = batch.get("image")
    if not isinstance(image, torch.Tensor):
        raise TypeError("batch.image must be a tensor")
    expected = (image.shape[0], 1, image.shape[-2], image.shape[-1])
    if tuple(valid_mask.shape) != tuple(expected):
        raise ValueError(
            "batch.valid_mask must be Bx1xHxW aligned with batch.image: "
            f"image={tuple(image.shape)} mask={tuple(valid_mask.shape)}"
        )
    return valid_mask.to(device=device, non_blocking=True)


def _batch_observer_image(
    batch: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor | None:
    """Move an optional native-resolution observer view to the model device."""

    observer_image = batch.get("observer_image")
    if observer_image is None:
        return None
    if not isinstance(observer_image, torch.Tensor):
        raise TypeError("batch.observer_image must be a tensor")
    image = batch.get("image")
    if not isinstance(image, torch.Tensor):
        raise TypeError("batch.image must be a tensor")
    if int(observer_image.shape[0]) != int(image.shape[0]):
        raise ValueError(
            "batch.observer_image must share batch size with batch.image: "
            f"image={tuple(image.shape)} observer={tuple(observer_image.shape)}"
        )
    return observer_image.to(device=device, non_blocking=True)


def _forward_surface_model(
    model: nn.Module,
    image: torch.Tensor,
    *,
    return_aux: bool,
    valid_mask: torch.Tensor | None,
    observer_image: torch.Tensor | None = None,
):
    """Forward a surface model, opting into masks only when one was emitted.

    No signature fallback is attempted.  If a config requests explicit masks
    for a wrapper/backbone that cannot consume them, failing closed is safer
    than silently reverting to RGB-based support inference.
    """

    if valid_mask is None and observer_image is None:
        return model(image, return_aux=return_aux)
    if observer_image is not None:
        if valid_mask is not None:
            raise ValueError(
                "observer_image and valid_mask cannot be combined by the "
                "current surface-model forwarding contract"
            )
        return model(
            image,
            return_aux=return_aux,
            observer_image=observer_image,
        )
    return model(
        image,
        return_aux=return_aux,
        valid_mask=valid_mask,
    )


def _inactive_loss_config_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _validate_linear_ce_fast_path(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    teacher_model: nn.Module | None,
    expert_teacher_model: nn.Module | None,
    anchor_teacher_logit_cache: dict[str, torch.Tensor] | None,
    expert_teacher_logit_cache: dict[str, torch.Tensor] | None,
    out_dir: Path | None = None,
) -> None:
    """Fail closed unless the generic objective is mathematically plain CE.

    The historical flag name predates structured classifiers.  The optimized
    loop does not assume that the classifier *head* is linear: it only asks the
    complete model for its final logits with ``return_aux=False`` and applies
    exactly the same cross-entropy, accumulation, clipping and optimizer step.
    Consequently a coupled/non-linear head is safe whenever every auxiliary
    objective is inactive and no teacher is attached.  Keeping the mathematical
    checks here, instead of checking ``head_type``, lets fair CE-only backbone
    comparisons skip dozens of zero-weight diagnostic computations without
    changing the model function or its gradient.
    """
    if any(
        item is not None
        for item in (
            teacher_model,
            expert_teacher_model,
            anchor_teacher_logit_cache,
            expert_teacher_logit_cache,
        )
    ):
        raise RuntimeError("train.linear_ce_fast_path cannot be used with teachers")
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    if bool(
        loss_cfg.get(
            "rscd_pcgrad_enabled",
            train_cfg.get("rscd_pcgrad_enabled", False),
        )
    ):
        raise RuntimeError("train.linear_ce_fast_path cannot be used with PCGrad")
    step_checkpoint_every = int(
        train_cfg.get("save_step_checkpoint_every", 0) or 0
    )
    if step_checkpoint_every < 0:
        raise RuntimeError(
            "train.linear_ce_fast_path requires "
            "save_step_checkpoint_every>=0"
        )
    resume_start_step = int(
        train_cfg.get(
            "_resume_start_step",
            train_cfg.get("resume_start_step", 0),
        )
        or 0
    )
    if resume_start_step < 0:
        raise RuntimeError(
            "train.linear_ce_fast_path requires resume_start_step>=0"
        )
    accumulation_steps = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    if resume_start_step > 0 and resume_start_step % accumulation_steps != 0:
        raise RuntimeError(
            "train.linear_ce_fast_path can resume only at an "
            "optimizer-update boundary: "
            f"step={resume_start_step} accum={accumulation_steps}"
        )
    ce_keys = {"label_smoothing", "focus_ce_extra_weight", "focus_ce_classes"}
    active_non_ce = {
        key: value
        for key, value in loss_cfg.items()
        if key not in ce_keys and not _inactive_loss_config_value(value)
    }
    if active_non_ce:
        raise RuntimeError(
            "train.linear_ce_fast_path found active non-CE loss settings: "
            f"{sorted(active_non_ce)}"
        )
    if step_checkpoint_every > 0 and out_dir is None:
        raise RuntimeError(
            "train.linear_ce_fast_path step checkpointing requires a "
            "concrete out_dir"
        )


def _train_one_epoch_linear_ce_fast(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    *,
    idx_to_class: dict[int, str],
    ema: ModelEMA | None,
    out_dir: Path | None = None,
    epoch: int = 1,
    class_to_idx: dict[str, int] | None = None,
    run_provenance: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Exact CE-only specialization of the generic training loop.

    The model forward, CE definition, accumulation, clipping, optimizer and EMA
    semantics are unchanged.  It only skips dozens of zero-weight auxiliary
    loss functions and their per-batch GPU-to-CPU diagnostic synchronizations.
    """

    _set_training_mode(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = _amp_autocast_dtype(train_cfg, device) if use_amp else None
    accum = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    resume_start_step = int(
        train_cfg.get(
            "_resume_start_step",
            train_cfg.get("resume_start_step", 0),
        )
        or 0
    )
    if resume_start_step < 0:
        raise RuntimeError("resume_start_step cannot be negative")
    if resume_start_step > 0 and resume_start_step % accum != 0:
        raise RuntimeError(
            "cannot resume CE fast path inside a gradient-accumulation window: "
            f"step={resume_start_step} accum={accum}"
        )
    resume_partial = (
        train_cfg.get("_resume_train_partial", {})
        if resume_start_step > 0
        else {}
    )
    if not isinstance(resume_partial, dict):
        raise RuntimeError("CE fast-path resume train_partial must be a mapping")
    total_seen = max(int(resume_partial.get("seen", 0) or 0), 0)
    total_loss = float(resume_partial.get("loss", 0.0) or 0.0) * total_seen
    if "correct" in resume_partial:
        total_correct = max(int(resume_partial.get("correct", 0) or 0), 0)
    else:
        total_correct = int(
            round(float(resume_partial.get("top1", 0.0) or 0.0) * total_seen)
        )
    nonfinite_grad_skips = max(
        int(resume_partial.get("nonfinite_grad_skips", 0) or 0),
        0,
    )
    last_nonfinite_grad_step = resume_partial.get("last_nonfinite_grad_step")
    skip_nonfinite_grad_steps, max_nonfinite_grad_skips = (
        _nonfinite_gradient_recovery_settings(train_cfg)
    )
    total_steps = resume_start_step + len(loader)
    step_checkpoint_every = int(
        train_cfg.get("save_step_checkpoint_every", 0) or 0
    )
    if step_checkpoint_every < 0:
        raise RuntimeError("save_step_checkpoint_every cannot be negative")
    if step_checkpoint_every > 0 and out_dir is None:
        raise RuntimeError(
            "CE fast-path step checkpointing requires a concrete out_dir"
        )
    last_step_checkpoint_step = resume_start_step
    log_every = int(train_cfg.get("log_every_steps", 80))
    for local_step, batch in enumerate(
        tqdm(loader, desc="train-ce-fast", leave=False, ascii=True),
        1,
    ):
        step = resume_start_step + local_step
        accumulation_window_size, optimizer_update_boundary = _accumulation_window(
            step,
            total_steps=total_steps,
            accumulation_steps=accum,
        )
        completed_updates_before_step = (
            _completed_optimizer_updates_before_step(
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                accumulation_steps=accum,
            )
        )
        _set_optimizer_update_curricula(
            model,
            completed_updates_before_step,
        )
        image = batch["image"].to(device, non_blocking=True)
        valid_mask = _batch_valid_mask(batch, device)
        observer_image = _batch_observer_image(batch, device)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            raw_output = _forward_surface_model(
                model,
                image,
                return_aux=False,
                valid_mask=valid_mask,
                observer_image=observer_image,
            )
            logits = (
                raw_output["logits"]
                if isinstance(raw_output, dict)
                else raw_output
            )
            loss = focus_weighted_cross_entropy(
                logits,
                label,
                idx_to_class,
                loss_cfg,
            )
            backward = loss / float(accumulation_window_size)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "CE fast-path loss is non-finite; refusing to skip the batch"
            )
        scaler.scale(backward).backward()
        if optimizer_update_boundary:
            scaler.unscale_(optimizer)
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_cfg.get("grad_clip_norm", 5.0)),
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                if (
                    not skip_nonfinite_grad_steps
                    or "non-finite" not in str(exc)
                ):
                    raise
                if nonfinite_grad_skips >= max_nonfinite_grad_skips:
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "CE fast-path encountered too many non-finite "
                        "gradient steps; refusing to continue silently: "
                        f"skips={nonfinite_grad_skips} "
                        f"max={max_nonfinite_grad_skips}"
                    ) from exc
                nonfinite_grad_skips += 1
                last_nonfinite_grad_step = int(step)
                optimizer.zero_grad(set_to_none=True)
                batch_size = int(label.numel())
                batch_loss = float(loss.detach().cpu())
                batch_correct = int(
                    (logits.argmax(dim=1) == label).sum().detach().cpu()
                )
                total_loss += batch_loss * batch_size
                total_correct += batch_correct
                total_seen += batch_size
                if out_dir is not None:
                    event = {
                        "epoch": int(epoch),
                        "step": int(step),
                        "total_steps": int(total_steps),
                        "batch_size": int(batch_size),
                        "batch_loss": batch_loss,
                        "batch_correct": int(batch_correct),
                        "nonfinite_grad_skips": int(nonfinite_grad_skips),
                        "max_nonfinite_grad_skips": int(
                            max_nonfinite_grad_skips
                        ),
                        "action": "skipped_optimizer_step",
                        "error": str(exc),
                    }
                    event.update(
                        _nonfinite_gradient_diagnostics(
                            model,
                            batch,
                            label,
                            idx_to_class,
                        )
                    )
                    with (out_dir / "nonfinite_gradient_events.jsonl").open(
                        "a",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(
                            json.dumps(event, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                print(
                    "  WARNING: non-finite gradient norm at "
                    f"step {step}/{total_steps}; skipped optimizer step "
                    f"({nonfinite_grad_skips}/{max_nonfinite_grad_skips})",
                    flush=True,
                )
                if log_every > 0 and (
                    step % log_every == 0 or local_step == len(loader)
                ):
                    print(
                        f"  train step {step}/{total_steps} "
                        f"loss={total_loss/max(total_seen,1):.4f} "
                        f"top1={total_correct/max(total_seen,1):.4f}"
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
            _set_optimizer_update_curricula(
                model,
                completed_updates_before_step + 1,
            )
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int(
            (logits.argmax(dim=1) == label).sum().detach().cpu()
        )
        total_seen += batch_size
        checkpoint_due = (
            step_checkpoint_every > 0
            and out_dir is not None
            and optimizer_update_boundary
            and (
                step - last_step_checkpoint_step >= step_checkpoint_every
                or local_step == len(loader)
            )
        )
        if checkpoint_due:
            step_state = _training_step_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader=loader,
                cfg=cfg,
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                class_to_idx=class_to_idx,
                run_provenance=run_provenance,
                train_partial={
                    "loss": total_loss / max(total_seen, 1),
                    "top1": total_correct / max(total_seen, 1),
                    "correct": int(total_correct),
                    "seen": int(total_seen),
                    "aux_log_sum": {"linear_ce_fast_path": 1.0},
                    "aux_log_count": 1,
                    "nonfinite_grad_skips": int(nonfinite_grad_skips),
                    "last_nonfinite_grad_step": last_nonfinite_grad_step,
                },
                ema=ema,
            )
            assert out_dir is not None
            _atomic_torch_save(
                step_state,
                out_dir / "last_step_checkpoint.pth",
            )
            last_step_checkpoint_step = int(step)
            print(
                "  saved step checkpoint: "
                f"{out_dir / 'last_step_checkpoint.pth'} "
                f"step={step}/{total_steps}"
            )
        if log_every > 0 and (
            step % log_every == 0 or local_step == len(loader)
        ):
            print(
                f"  train step {step}/{total_steps} "
                f"loss={total_loss/max(total_seen,1):.4f} "
                f"top1={total_correct/max(total_seen,1):.4f}"
            )
    result = {
        "loss": total_loss / max(total_seen, 1),
        "top1": total_correct / max(total_seen, 1),
        "linear_ce_fast_path": 1.0,
        "nonfinite_grad_skips": int(nonfinite_grad_skips),
    }
    if last_nonfinite_grad_step is not None:
        result["last_nonfinite_grad_step"] = int(last_nonfinite_grad_step)
    result.update(_optimizer_update_curriculum_logs(model))
    return result


_CACHED_TEACHER_FAST_LOSS_KEYS = {
    "label_smoothing",
    "focus_ce_extra_weight",
    "focus_ce_classes",
    "factor_weight",
    "factor_axis_weights",
    "supervise_none",
    "cached_teacher_distill_weight",
    "cached_teacher_distill_temperature",
    "cached_teacher_distill_confidence",
    "cached_teacher_distill_correct_only",
    "cached_teacher_distill_class_balanced",
    "cached_teacher_relation_distill_weight",
    "cached_teacher_relation_distill_representation_key",
    "cached_teacher_relation_distill_student_key",
    "cached_teacher_relation_distill_beta",
    "cached_teacher_relation_distill_eps",
    "cached_teacher_coordinate_distill_weight",
    "cached_teacher_coordinate_distill_representation_key",
    "cached_teacher_coordinate_distill_student_key",
    "cached_teacher_coordinate_distill_beta",
    "cached_teacher_coordinate_distill_cosine_weight",
    "cached_teacher_coordinate_distill_eps",
    "facet_joint_distill_weight",
    "facet_joint_distill_temperature",
    "facc_node_ce_weight_stage3",
    "facc_node_ce_weight_stage4",
    "progressive_shared_head_weights",
}


def _facc_node_auxiliary_loss(
    model_output: Mapping[str, Any],
    labels: torch.Tensor,
    loss_cfg: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Lock FACC's legal-state node semantics with shared-reader CE losses."""

    logits = model_output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("FACC node auxiliary loss requires tensor logits")
    weights = {
        "stage3": float(loss_cfg.get("facc_node_ce_weight_stage3", 0.0)),
        "stage4": float(loss_cfg.get("facc_node_ce_weight_stage4", 0.0)),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("FACC node CE weights must be finite and non-negative")
    if not any(value > 0.0 for value in weights.values()):
        return logits.sum() * 0.0, {
            "loss_facc_node_aux": 0.0,
            "loss_facc_node_stage3": 0.0,
            "loss_facc_node_stage4": 0.0,
        }

    node_logits = model_output.get("facc_node_logits")
    if not isinstance(node_logits, Mapping):
        raise RuntimeError(
            "positive FACC node CE weight requires output['facc_node_logits']"
        )
    terms: list[torch.Tensor] = []
    logs: dict[str, float] = {}
    for stage_name, weight in weights.items():
        if weight <= 0.0:
            logs[f"loss_facc_node_{stage_name}"] = 0.0
            continue
        stage_logits = node_logits.get(stage_name)
        if not isinstance(stage_logits, torch.Tensor):
            raise RuntimeError(
                f"FACC node output is missing tensor {stage_name!r}"
            )
        if tuple(stage_logits.shape) != tuple(logits.shape):
            raise RuntimeError(
                f"FACC {stage_name} logits must match deployed logits shape: "
                f"{tuple(stage_logits.shape)} versus {tuple(logits.shape)}"
            )
        raw = F.cross_entropy(stage_logits.float(), labels, reduction="mean")
        weighted = float(weight) * raw
        terms.append(weighted)
        logs[f"loss_facc_node_{stage_name}"] = float(weighted.detach().cpu())
        logs[f"facc_node_{stage_name}_top1"] = float(
            stage_logits.detach().argmax(dim=1).eq(labels).float().mean().cpu()
        )
    loss = torch.stack(terms).sum().to(dtype=logits.dtype)
    logs["loss_facc_node_aux"] = float(loss.detach().cpu())
    return loss, logs


def _progressive_shared_head_weights(
    loss_cfg: Mapping[str, Any],
    *,
    expected_count: int | None = None,
) -> tuple[float, ...] | None:
    raw_weights = loss_cfg.get("progressive_shared_head_weights")
    if raw_weights is None:
        return None
    if not isinstance(raw_weights, (list, tuple)) or not raw_weights:
        raise ValueError(
            "loss.progressive_shared_head_weights must be a non-empty sequence"
        )
    weights = tuple(float(value) for value in raw_weights)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(
            "loss.progressive_shared_head_weights must be finite and non-negative"
        )
    if abs(sum(weights) - 1.0) > 1.0e-6:
        raise ValueError(
            "loss.progressive_shared_head_weights must sum to 1; implicit "
            "normalization is forbidden"
        )
    if expected_count is not None and len(weights) != int(expected_count):
        raise ValueError(
            "progressive logits/weight count mismatch: "
            f"logits={expected_count} weights={len(weights)}"
        )
    return weights


def _cached_teacher_selection_mask(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: Mapping[str, Any],
) -> torch.Tensor:
    """Compute the correctness/confidence mask once for every shared prefix."""

    confidence_threshold = min(
        max(float(loss_cfg.get("cached_teacher_distill_confidence", 0.0)), 0.0),
        1.0,
    )
    correct_only = bool(
        loss_cfg.get("cached_teacher_distill_correct_only", True)
    )
    with torch.no_grad(), torch.amp.autocast(
        device_type=teacher_logits.device.type,
        enabled=False,
    ):
        teacher_probability = F.softmax(teacher_logits.detach().float(), dim=1)
        confidence, prediction = teacher_probability.max(dim=1)
        selected = confidence.ge(confidence_threshold)
        if correct_only:
            selected &= prediction.eq(labels)
    return selected


def _cached_teacher_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: dict[str, Any],
    *,
    selected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generic, teacher-correct distillation for deterministic pre-warming.

    The released teacher is deliberately not trusted on every RSCD example.
    By default, only samples on which its top-1 agrees with the ground-truth
    label contribute a soft-target term.  The term is class-balanced so the
    high-support composite classes cannot dominate the pre-warm.  A later
    CE-only curriculum stage removes this constraint entirely, allowing the
    student to exceed the teacher rather than inheriting its error ceiling.
    """

    weight = float(loss_cfg.get("cached_teacher_distill_weight", 0.0))
    if weight <= 0.0:
        return student_logits.new_zeros(()), {
            "loss_cached_teacher_distill": 0.0,
            "cached_teacher_distill_count": 0.0,
            "cached_teacher_distill_active_classes": 0.0,
        }
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "cached teacher logits must match student logits: "
            f"student={tuple(student_logits.shape)} "
            f"teacher={tuple(teacher_logits.shape)}"
        )
    temperature = max(
        float(loss_cfg.get("cached_teacher_distill_temperature", 2.0)),
        1.0e-3,
    )
    confidence_threshold = min(
        max(float(loss_cfg.get("cached_teacher_distill_confidence", 0.0)), 0.0),
        1.0,
    )
    correct_only = bool(
        loss_cfg.get("cached_teacher_distill_correct_only", True)
    )
    class_balanced = bool(
        loss_cfg.get("cached_teacher_distill_class_balanced", True)
    )

    with torch.amp.autocast(
        device_type=student_logits.device.type,
        enabled=False,
    ):
        student_fp32 = student_logits.float()
        teacher_fp32 = teacher_logits.detach().float()
        teacher_prob = F.softmax(teacher_fp32, dim=1)
        teacher_confidence, teacher_prediction = teacher_prob.max(dim=1)
        if selected_mask is None:
            selected = teacher_confidence.ge(confidence_threshold)
            if correct_only:
                selected &= teacher_prediction.eq(labels)
        else:
            if tuple(selected_mask.shape) != tuple(labels.shape):
                raise ValueError("cached teacher selected_mask must match labels")
            selected = selected_mask.to(
                device=student_logits.device,
                dtype=torch.bool,
            )
        per_sample = F.kl_div(
            F.log_softmax(student_fp32 / temperature, dim=1),
            F.softmax(teacher_fp32 / temperature, dim=1),
            reduction="none",
        ).sum(dim=1).clamp_min(0.0) * (temperature * temperature)

        terms: list[torch.Tensor] = []
        active_classes = 0
        if class_balanced:
            for class_index in labels.detach().unique().tolist():
                class_mask = selected & labels.eq(int(class_index))
                if bool(class_mask.any()):
                    terms.append(per_sample[class_mask].mean())
                    active_classes += 1
        elif bool(selected.any()):
            terms.append(per_sample[selected].mean())
            active_classes = int(labels[selected].detach().unique().numel())

    if not terms:
        return student_logits.new_zeros(()), {
            "loss_cached_teacher_distill": 0.0,
            "cached_teacher_distill_count": 0.0,
            "cached_teacher_distill_active_classes": 0.0,
            "cached_teacher_confidence_mean": float(
                teacher_confidence.detach().mean().cpu()
            ),
        }
    raw_loss = torch.stack(terms).mean()
    loss = (weight * raw_loss).to(dtype=student_logits.dtype)
    return loss, {
        "loss_cached_teacher_distill": float(loss.detach().cpu()),
        "cached_teacher_distill_raw": float(raw_loss.detach().cpu()),
        "cached_teacher_distill_count": float(selected.sum().detach().cpu()),
        "cached_teacher_distill_active_classes": float(active_classes),
        "cached_teacher_confidence_mean": float(
            teacher_confidence.detach().mean().cpu()
        ),
        "cached_teacher_correct_rate": float(
            teacher_prediction.eq(labels).float().detach().mean().cpu()
        ),
    }


def _facet_joint_distillation_loss(
    student_joint_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    class_to_factor: torch.Tensor,
    loss_cfg: Mapping[str, Any],
    *,
    selected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Distill RSCD-27 teacher mass into FACET's complete 6x5x4 lattice.

    The 27-way temperature distribution is scattered onto its labelled cells;
    the other cells remain finite competitors with zero target mass.  This
    anchors FACET's axes without putting a legal-state mask or a class-specific
    route inside the deployed backbone.
    """

    weight = float(loss_cfg.get("facet_joint_distill_weight", 0.0))
    empty_logs = {
        "loss_facet_joint_distill": 0.0,
        "facet_joint_distill_count": 0.0,
        "trace_facet_valid_mass_mean": 0.0,
        "trace_facet_valid_cond_kl_mean": 0.0,
        "trace_facet_full120_kl_mean": 0.0,
        "trace_facet_teacher_top1_full120": 0.0,
        "trace_facet_teacher_top1_valid": 0.0,
    }
    if weight <= 0.0:
        return student_joint_logits.sum() * 0.0, empty_logs
    if student_joint_logits.ndim != 2 or int(student_joint_logits.shape[1]) != 120:
        raise ValueError(
            "FACET joint logits must have shape Bx120, got "
            f"{tuple(student_joint_logits.shape)}"
        )
    batch_size = int(student_joint_logits.shape[0])
    if teacher_logits.ndim != 2 or tuple(teacher_logits.shape) != (
        batch_size,
        int(class_to_factor.shape[0]),
    ):
        raise ValueError(
            "FACET teacher/class-factor shape mismatch: "
            f"student={tuple(student_joint_logits.shape)} "
            f"teacher={tuple(teacher_logits.shape)} "
            f"class_to_factor={tuple(class_to_factor.shape)}"
        )
    if labels.ndim != 1 or int(labels.shape[0]) != batch_size:
        raise ValueError("FACET labels must have shape B")
    factors = class_to_factor.to(
        device=student_joint_logits.device,
        dtype=torch.long,
    )
    if factors.ndim != 2 or int(factors.shape[1]) != 3 or bool(factors.lt(0).any()):
        raise ValueError("FACET requires a complete non-negative class_to_factor table")
    if bool(
        (factors[:, 0].ge(6) | factors[:, 1].ge(5) | factors[:, 2].ge(4)).any()
    ):
        raise ValueError("FACET class_to_factor exceeds the fixed 6x5x4 lattice")
    flat_class_cells = factors[:, 0] * 20 + factors[:, 1] * 4 + factors[:, 2]
    if int(flat_class_cells.unique().numel()) != int(flat_class_cells.numel()):
        raise ValueError("FACET classes must map to unique lattice cells")

    if selected_mask is None:
        selected = torch.ones(
            batch_size,
            device=student_joint_logits.device,
            dtype=torch.bool,
        )
    else:
        selected = selected_mask.to(
            device=student_joint_logits.device,
            dtype=torch.bool,
        )
        if tuple(selected.shape) != (batch_size,):
            raise ValueError("FACET selected_mask must have shape B")
    temperature = max(
        float(loss_cfg.get("facet_joint_distill_temperature", 2.0)),
        1.0e-3,
    )
    with torch.amp.autocast(
        device_type=student_joint_logits.device.type,
        enabled=False,
    ):
        teacher_probability = F.softmax(
            teacher_logits.detach().float() / temperature,
            dim=1,
        )
        scaled_all_logits = student_joint_logits.float() / temperature
        valid_logits = scaled_all_logits.index_select(1, flat_class_cells)
        valid_log_probability = F.log_softmax(
            valid_logits,
            dim=1,
        )
        conditional_kl = F.kl_div(
            valid_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=1).clamp_min(0.0)
        log_valid_mass = torch.logsumexp(valid_logits, dim=1) - torch.logsumexp(
            scaled_all_logits,
            dim=1,
        )
        valid_mass_penalty = -log_valid_mass
        full120_kl = conditional_kl + valid_mass_penalty
        scale = temperature * temperature
        raw_loss = (
            (scale * full120_kl[selected]).mean()
            if bool(selected.any())
            else student_joint_logits.float().sum() * 0.0
        )

        with torch.no_grad():
            report_mask = selected if bool(selected.any()) else torch.ones_like(selected)
            valid_mass = log_valid_mass.exp()
            teacher_prediction = teacher_logits.detach().argmax(dim=1)
            teacher_cells = flat_class_cells.index_select(
                0, teacher_prediction.to(device=flat_class_cells.device)
            )
            full_agreement = student_joint_logits.detach().argmax(dim=1).eq(teacher_cells)
            valid_agreement = valid_logits.detach().argmax(dim=1).eq(teacher_prediction)
    loss = (weight * raw_loss).to(dtype=student_joint_logits.dtype)
    return loss, {
        "loss_facet_joint_distill": float(loss.detach().cpu()),
        "facet_joint_distill_count": float(selected.sum().detach().cpu()),
        "trace_facet_valid_mass_mean": float(
            valid_mass[report_mask].mean().detach().cpu()
        ),
        "trace_facet_valid_cond_kl_mean": float(
            (scale * conditional_kl[report_mask]).mean().detach().cpu()
        ),
        "trace_facet_full120_kl_mean": float(
            (scale * full120_kl[report_mask]).mean().detach().cpu()
        ),
        "trace_facet_teacher_top1_full120": float(
            full_agreement[report_mask].float().mean().detach().cpu()
        ),
        "trace_facet_teacher_top1_valid": float(
            valid_agreement[report_mask].float().mean().detach().cpu()
        ),
    }


def _cached_teacher_relation_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: Mapping[str, Any],
    *,
    selected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Basis-insensitive relation distillation from an offline embedding.

    Direct coordinate MSE is invalid when independently parameterized teacher
    and student backbones use unrelated channel bases.  Instead, this compares
    the upper triangle of their sample-by-sample cosine Gram matrices.  Only
    rows whose cached ensemble prediction is correct are eligible.  Fewer than
    two eligible rows contain no pairwise relation and therefore return a
    differentiable zero.
    """

    weight = float(loss_cfg.get("cached_teacher_relation_distill_weight", 0.0))
    zero = student_embedding.sum() * 0.0
    empty_logs = {
        "loss_cached_teacher_relation_distill": 0.0,
        "cached_teacher_relation_selected_count": 0.0,
        "cached_teacher_relation_pair_count": 0.0,
    }
    if weight <= 0.0:
        return zero, empty_logs
    if student_embedding.ndim != 2 or teacher_embedding.ndim != 2:
        raise ValueError(
            "cached relation distillation requires rank-2 student and teacher "
            "embeddings"
        )
    if int(student_embedding.shape[0]) != int(labels.numel()) or int(
        teacher_embedding.shape[0]
    ) != int(labels.numel()):
        raise ValueError(
            "cached relation embeddings and labels must have the same batch size"
        )
    if int(student_embedding.shape[1]) <= 0 or int(teacher_embedding.shape[1]) <= 0:
        raise ValueError("cached relation embedding widths must be positive")
    if tuple(teacher_logits.shape[:1]) != tuple(labels.shape):
        raise ValueError("cached teacher logits and labels must align")
    if not bool(torch.isfinite(student_embedding.detach()).all()) or not bool(
        torch.isfinite(teacher_embedding.detach()).all()
    ):
        raise FloatingPointError(
            "cached relation distillation received NaN or Inf embeddings"
        )
    if selected_mask is None:
        selected = _cached_teacher_selection_mask(
            teacher_logits,
            labels,
            loss_cfg,
        )
    else:
        if tuple(selected_mask.shape) != tuple(labels.shape):
            raise ValueError("cached relation selected_mask must match labels")
        selected = selected_mask.to(
            device=student_embedding.device,
            dtype=torch.bool,
        )
    selected_count = int(selected.sum().detach().cpu())
    if selected_count < 2:
        empty_logs["cached_teacher_relation_selected_count"] = float(
            selected_count
        )
        return zero, empty_logs

    beta = float(loss_cfg.get("cached_teacher_relation_distill_beta", 0.1))
    eps = float(loss_cfg.get("cached_teacher_relation_distill_eps", 1.0e-8))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("cached_teacher_relation_distill_beta must be positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("cached_teacher_relation_distill_eps must be positive")
    with torch.amp.autocast(
        device_type=student_embedding.device.type,
        enabled=False,
    ):
        student_selected = F.normalize(
            student_embedding.float()[selected],
            p=2.0,
            dim=1,
            eps=eps,
        )
        teacher_selected = F.normalize(
            teacher_embedding.detach().float()[selected],
            p=2.0,
            dim=1,
            eps=eps,
        )
        student_gram = student_selected @ student_selected.transpose(0, 1)
        teacher_gram = teacher_selected @ teacher_selected.transpose(0, 1)
        pair_indices = torch.triu_indices(
            selected_count,
            selected_count,
            offset=1,
            device=student_embedding.device,
        )
        student_pairs = student_gram[pair_indices[0], pair_indices[1]]
        teacher_pairs = teacher_gram[pair_indices[0], pair_indices[1]]
        raw_loss = F.smooth_l1_loss(
            student_pairs,
            teacher_pairs,
            reduction="mean",
            beta=beta,
        )
        mean_abs_gap = (student_pairs - teacher_pairs).abs().mean()
    if not bool(torch.isfinite(raw_loss.detach())):
        raise FloatingPointError(
            "cached teacher relation distillation produced NaN or Inf"
        )
    loss = (weight * raw_loss).to(dtype=student_embedding.dtype)
    return loss, {
        "loss_cached_teacher_relation_distill": float(loss.detach().cpu()),
        "cached_teacher_relation_distill_raw": float(raw_loss.detach().cpu()),
        "cached_teacher_relation_selected_count": float(selected_count),
        "cached_teacher_relation_pair_count": float(student_pairs.numel()),
        "cached_teacher_relation_student_cosine_mean": float(
            student_pairs.detach().mean().cpu()
        ),
        "cached_teacher_relation_teacher_cosine_mean": float(
            teacher_pairs.detach().mean().cpu()
        ),
        "cached_teacher_relation_mean_abs_gap": float(
            mean_abs_gap.detach().cpu()
        ),
    }


def _cached_teacher_coordinate_distillation_loss(
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_cfg: Mapping[str, Any],
    *,
    selected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Recover an explicitly shared teacher coordinate system.

    This objective is intentionally stricter than relation distillation.  It
    is admitted only when student and teacher widths are identical.  TPDR uses
    it after inheriting the S7 projection and complete downstream C3 head, so
    every coordinate has a fixed operational meaning.  Independently
    parameterized backbones should keep this weight at zero and use a separate
    audited projector or a basis-insensitive objective instead.
    """

    weight = float(loss_cfg.get("cached_teacher_coordinate_distill_weight", 0.0))
    zero = student_embedding.sum() * 0.0
    empty_logs = {
        "loss_cached_teacher_coordinate_distill": 0.0,
        "cached_teacher_coordinate_selected_count": 0.0,
    }
    if weight <= 0.0:
        return zero, empty_logs
    if student_embedding.ndim != 2 or teacher_embedding.ndim != 2:
        raise ValueError(
            "cached coordinate distillation requires rank-2 student and "
            "teacher embeddings"
        )
    if tuple(student_embedding.shape) != tuple(teacher_embedding.shape):
        raise ValueError(
            "cached coordinate distillation requires identical student and "
            "teacher shapes; received "
            f"{tuple(student_embedding.shape)} and {tuple(teacher_embedding.shape)}"
        )
    if int(student_embedding.shape[0]) != int(labels.numel()):
        raise ValueError("cached coordinate embeddings and labels must align")
    if tuple(teacher_logits.shape[:1]) != tuple(labels.shape):
        raise ValueError("cached teacher logits and labels must align")
    if not bool(torch.isfinite(student_embedding.detach()).all()) or not bool(
        torch.isfinite(teacher_embedding.detach()).all()
    ):
        raise FloatingPointError(
            "cached coordinate distillation received NaN or Inf embeddings"
        )
    if selected_mask is None:
        selected = _cached_teacher_selection_mask(
            teacher_logits,
            labels,
            loss_cfg,
        )
    else:
        if tuple(selected_mask.shape) != tuple(labels.shape):
            raise ValueError("cached coordinate selected_mask must match labels")
        selected = selected_mask.to(
            device=student_embedding.device,
            dtype=torch.bool,
        )
    selected_count = int(selected.sum().detach().cpu())
    if selected_count == 0:
        return zero, empty_logs

    beta = float(loss_cfg.get("cached_teacher_coordinate_distill_beta", 0.1))
    cosine_weight = float(
        loss_cfg.get("cached_teacher_coordinate_distill_cosine_weight", 1.0)
    )
    eps = float(loss_cfg.get("cached_teacher_coordinate_distill_eps", 1.0e-8))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("cached_teacher_coordinate_distill_beta must be positive")
    if not math.isfinite(cosine_weight) or cosine_weight < 0.0:
        raise ValueError(
            "cached_teacher_coordinate_distill_cosine_weight must be "
            "finite and non-negative"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("cached_teacher_coordinate_distill_eps must be positive")

    with torch.amp.autocast(
        device_type=student_embedding.device.type,
        enabled=False,
    ):
        student_selected = student_embedding.float()[selected]
        teacher_selected = teacher_embedding.detach().float()[selected]
        huber = F.smooth_l1_loss(
            student_selected,
            teacher_selected,
            reduction="mean",
            beta=beta,
        )
        cosine = F.cosine_similarity(
            student_selected,
            teacher_selected,
            dim=1,
            eps=eps,
        )
        cosine_gap = (1.0 - cosine).mean()
        raw_loss = huber + cosine_weight * cosine_gap
        difference_norm = torch.linalg.vector_norm(
            student_selected - teacher_selected,
            dim=1,
        )
        teacher_norm = torch.linalg.vector_norm(
            teacher_selected,
            dim=1,
        ).clamp_min(eps)
        relative_error = (difference_norm / teacher_norm).mean()
    if not bool(torch.isfinite(raw_loss.detach())):
        raise FloatingPointError(
            "cached teacher coordinate distillation produced NaN or Inf"
        )
    loss = (weight * raw_loss).to(dtype=student_embedding.dtype)
    return loss, {
        "loss_cached_teacher_coordinate_distill": float(loss.detach().cpu()),
        "cached_teacher_coordinate_distill_raw": float(raw_loss.detach().cpu()),
        "cached_teacher_coordinate_huber": float(huber.detach().cpu()),
        "cached_teacher_coordinate_cosine_mean": float(cosine.mean().detach().cpu()),
        "cached_teacher_coordinate_relative_error": float(
            relative_error.detach().cpu()
        ),
        "cached_teacher_coordinate_selected_count": float(selected_count),
    }


def _cached_teacher_progressive_objective(
    model_output: Mapping[str, Any],
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    idx_to_class: dict[int, str],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """CE + correctness-masked KD for one shared head over ordered prefixes.

    The output contract is deliberately explicit.  A model must provide a
    non-empty ``progressive_logits`` tuple whose final object is the deployed
    ``logits`` tensor.  Repeating final logits or silently dropping prefixes is
    not accepted.
    """

    final_logits = model_output.get("logits")
    progressive_logits = model_output.get("progressive_logits")
    if not isinstance(final_logits, torch.Tensor):
        raise RuntimeError("progressive cached-KD requires tensor output['logits']")
    if not isinstance(progressive_logits, (list, tuple)) or not progressive_logits:
        raise RuntimeError(
            "progressive cached-KD requires a non-empty output['progressive_logits']"
        )
    prefixes = tuple(progressive_logits)
    weights = _progressive_shared_head_weights(
        loss_cfg,
        expected_count=len(prefixes),
    )
    if weights is None:  # guarded by the caller, retained as a fail-closed API
        raise RuntimeError("progressive cached-KD weights are not configured")
    if prefixes[-1] is not final_logits:
        raise RuntimeError(
            "progressive_logits[-1] must be the exact deployed logits tensor"
        )
    for index, prefix_logits in enumerate(prefixes):
        if not isinstance(prefix_logits, torch.Tensor):
            raise TypeError(f"progressive_logits[{index}] must be a tensor")
        if tuple(prefix_logits.shape) != tuple(final_logits.shape):
            raise ValueError(
                f"progressive_logits[{index}] shape differs from final logits"
            )
        if not bool(torch.isfinite(prefix_logits.detach()).all()):
            raise FloatingPointError(
                f"progressive_logits[{index}] contains NaN or Inf"
            )

    selection_mask = _cached_teacher_selection_mask(
        teacher_logits,
        labels,
        loss_cfg,
    )
    ce_terms: list[torch.Tensor] = []
    kd_terms: list[torch.Tensor] = []
    first_kd_logs: dict[str, float] | None = None
    for weight, prefix_logits in zip(weights, prefixes, strict=True):
        if weight == 0.0:
            continue
        ce_terms.append(
            float(weight)
            * focus_weighted_cross_entropy(
                prefix_logits,
                labels,
                idx_to_class,
                loss_cfg,
            )
        )
        prefix_kd, prefix_logs = _cached_teacher_distillation_loss(
            prefix_logits,
            teacher_logits,
            labels,
            loss_cfg,
            selected_mask=selection_mask,
        )
        kd_terms.append(float(weight) * prefix_kd)
        if first_kd_logs is None:
            first_kd_logs = prefix_logs
    if not ce_terms:
        raise ValueError("progressive shared-head weights must contain a positive value")
    main_loss = torch.stack(ce_terms).sum()
    distill_loss = (
        torch.stack(kd_terms).sum()
        if kd_terms
        else final_logits.new_zeros(())
    )
    logs = dict(first_kd_logs or {})
    logs["loss_cached_teacher_distill"] = float(distill_loss.detach().cpu())
    logs["progressive_shared_head_count"] = float(len(prefixes))
    logs["progressive_shared_head_weight_sum"] = float(sum(weights))
    logs["progressive_teacher_selected_count"] = float(
        selection_mask.sum().detach().cpu()
    )
    return final_logits, main_loss, distill_loss, logs


def _validate_cached_teacher_fast_path(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    teacher_model: nn.Module | None,
    expert_teacher_model: nn.Module | None,
    anchor_teacher_logit_cache: dict[str, torch.Tensor] | None,
    expert_teacher_logit_cache: dict[str, torch.Tensor] | None,
    teacher_cache_strict: bool,
) -> None:
    """Fail closed for deterministic CE + cached-teacher pre-warming."""

    head_type = str(getattr(model, "head_type", "")).strip().lower()
    structured_contract = str(
        cfg.get("train", {}).get(
            "cached_teacher_structured_head_contract",
            "",
        )
    ).strip()
    model_contract = str(
        getattr(model, "cached_teacher_fast_path_contract", "")
    ).strip()
    structured_head = head_type != "linear"
    if structured_head and not (
        structured_contract == "c3_structured_v1"
        and model_contract == structured_contract
    ):
        raise RuntimeError(
            "a structured cached-teacher run requires the exact mutually "
            "declared c3_structured_v1 contract"
        )
    if structured_head:
        train_cfg = cfg.get("train", {})
        data_cfg = cfg.get("data", {})
        if data_cfg.get("test_manifest"):
            raise RuntimeError(
                "c3_structured_v1 recovery must not configure a formal-test "
                "manifest"
            )
        if not bool(train_cfg.get("freeze_nontrainable_modules_eval", False)):
            raise RuntimeError(
                "c3_structured_v1 recovery requires frozen stochastic modules "
                "to remain in eval mode"
            )
        if not train_cfg.get("resume_from") or not train_cfg.get(
            "resume_expected_sha256"
        ):
            raise RuntimeError(
                "c3_structured_v1 recovery requires a pinned resume checkpoint"
            )
    if anchor_teacher_logit_cache is None:
        raise RuntimeError(
            "train.cached_teacher_fast_path requires teacher_logits_cache"
        )
    if teacher_model is not None:
        raise RuntimeError(
            "train.cached_teacher_fast_path cannot use an online teacher"
        )
    if expert_teacher_model is not None or expert_teacher_logit_cache is not None:
        raise RuntimeError(
            "train.cached_teacher_fast_path does not support specialist teachers"
        )
    if not teacher_cache_strict:
        raise RuntimeError(
            "train.cached_teacher_fast_path requires teacher_logits_cache_strict=true"
        )
    train_cfg = cfg.get("train", {})
    if bool(train_cfg.get("augmentation", True)):
        raise RuntimeError(
            "train.cached_teacher_fast_path requires augmentation=false because "
            "an image-path cache represents one deterministic teacher view"
        )
    if bool(train_cfg.get("teacher_cache_online_fallback", False)):
        raise RuntimeError(
            "train.cached_teacher_fast_path requires teacher_cache_online_fallback=false"
        )
    loss_cfg = cfg.get("loss", {})
    try:
        factor_weight = float(loss_cfg.get("factor_weight", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "loss.factor_weight must be finite and non-negative"
        ) from exc
    if not math.isfinite(factor_weight) or factor_weight < 0.0:
        raise RuntimeError(
            "loss.factor_weight must be finite and non-negative"
        )
    factor_axis_weights_raw = loss_cfg.get("factor_axis_weights")
    if factor_axis_weights_raw is None:
        factor_axis_weights = {axis: 1.0 for axis in FACTOR_AXES}
    elif not isinstance(factor_axis_weights_raw, Mapping):
        raise RuntimeError("loss.factor_axis_weights must be a mapping")
    else:
        unknown_factor_axes = set(factor_axis_weights_raw) - set(FACTOR_AXES)
        if unknown_factor_axes:
            raise RuntimeError(
                "loss.factor_axis_weights contains unknown axes: "
                f"{sorted(str(axis) for axis in unknown_factor_axes)}"
            )
        try:
            configured_factor_axis_weights = {
                str(axis): float(value)
                for axis, value in factor_axis_weights_raw.items()
            }
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "loss.factor_axis_weights values must be finite and non-negative"
            ) from exc
        factor_axis_weights = {
            axis: configured_factor_axis_weights.get(axis, 1.0)
            for axis in FACTOR_AXES
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in factor_axis_weights.values()
        ):
            raise RuntimeError(
                "loss.factor_axis_weights values must be finite and non-negative"
            )
    if "supervise_none" in loss_cfg and not isinstance(
        loss_cfg["supervise_none"], bool
    ):
        raise RuntimeError("loss.supervise_none must be a boolean")
    factor_aux_active = factor_weight > 0.0
    if factor_aux_active:
        if not any(value > 0.0 for value in factor_axis_weights.values()):
            raise RuntimeError(
                "positive loss.factor_weight requires at least one positive "
                "factor-axis weight"
            )
        backbone = getattr(model, "backbone", None)
        if not hasattr(backbone, "last_factor_logits"):
            raise RuntimeError(
                "positive loss.factor_weight requires a backbone exposing "
                "last_factor_logits"
            )
        spec = getattr(model, "spec", None)
        class_to_factor = getattr(spec, "class_to_factor", None)
        if not isinstance(class_to_factor, torch.Tensor) or tuple(
            class_to_factor.shape
        ) != (27, 3):
            raise RuntimeError(
                "factor auxiliary supervision requires an RSCD-27 "
                "class_to_factor tensor with shape 27x3"
            )
    logit_distill_weight = float(
        loss_cfg.get("cached_teacher_distill_weight", 0.0)
    )
    relation_distill_weight = float(
        loss_cfg.get("cached_teacher_relation_distill_weight", 0.0)
    )
    coordinate_distill_weight = float(
        loss_cfg.get("cached_teacher_coordinate_distill_weight", 0.0)
    )
    facet_joint_weight = float(
        loss_cfg.get("facet_joint_distill_weight", 0.0)
    )
    facc_node_weights = (
        float(loss_cfg.get("facc_node_ce_weight_stage3", 0.0)),
        float(loss_cfg.get("facc_node_ce_weight_stage4", 0.0)),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in facc_node_weights):
        raise RuntimeError("FACC node CE weights must be finite and non-negative")
    if any(value > 0.0 for value in facc_node_weights):
        backbone = getattr(model, "backbone", None)
        if not hasattr(backbone, "last_node_logits"):
            raise RuntimeError(
                "positive FACC node CE weight requires a backbone exposing "
                "last_node_logits"
            )
    facc_node_aux_active = any(
        float(loss_cfg.get(key, 0.0)) > 0.0
        for key in (
            "facc_node_ce_weight_stage3",
            "facc_node_ce_weight_stage4",
        )
    )
    if (
        logit_distill_weight <= 0.0
        and relation_distill_weight <= 0.0
        and coordinate_distill_weight <= 0.0
        and facet_joint_weight <= 0.0
    ):
        raise RuntimeError(
            "train.cached_teacher_fast_path requires at least one positive "
            "cached teacher logit, relation, coordinate, or FACET "
            "joint-distillation weight"
        )
    if facet_joint_weight > 0.0:
        backbone = getattr(model, "backbone", None)
        if not hasattr(backbone, "last_joint_logits"):
            raise RuntimeError(
                "facet_joint_distill_weight > 0 requires a backbone exposing "
                "last_joint_logits"
            )
        spec = getattr(model, "spec", None)
        class_to_factor = getattr(spec, "class_to_factor", None)
        if not isinstance(class_to_factor, torch.Tensor) or tuple(
            class_to_factor.shape
        ) != (27, 3):
            raise RuntimeError(
                "FACET joint distillation requires an RSCD-27 class_to_factor "
                "tensor with shape 27x3"
            )
        facet_temperature = float(
            loss_cfg.get("facet_joint_distill_temperature", 2.0)
        )
        if not math.isfinite(facet_temperature) or facet_temperature <= 0.0:
            raise RuntimeError(
                "facet_joint_distill_temperature must be finite and positive"
            )
    if relation_distill_weight > 0.0:
        if not bool(loss_cfg.get("cached_teacher_distill_correct_only", True)):
            raise RuntimeError(
                "cached teacher relation distillation requires "
                "cached_teacher_distill_correct_only=true"
            )
        representation_key = str(
            loss_cfg.get(
                "cached_teacher_relation_distill_representation_key",
                "backbone_embedding",
            )
        ).strip()
        if not representation_key:
            raise RuntimeError(
                "cached teacher relation representation key must be non-empty"
            )
        if not isinstance(anchor_teacher_logit_cache, TeacherLogitCache) or (
            representation_key
            not in anchor_teacher_logit_cache.representation_rows
        ):
            raise RuntimeError(
                "cached teacher relation distillation requires a strictly "
                f"loaded representation {representation_key!r}"
            )
    if coordinate_distill_weight > 0.0:
        if not bool(loss_cfg.get("cached_teacher_distill_correct_only", True)):
            raise RuntimeError(
                "cached teacher coordinate distillation requires "
                "cached_teacher_distill_correct_only=true"
            )
        representation_key = str(
            loss_cfg.get(
                "cached_teacher_coordinate_distill_representation_key",
                "backbone_embedding",
            )
        ).strip()
        if not representation_key:
            raise RuntimeError(
                "cached teacher coordinate representation key must be non-empty"
            )
        if not isinstance(anchor_teacher_logit_cache, TeacherLogitCache) or (
            representation_key
            not in anchor_teacher_logit_cache.representation_rows
        ):
            raise RuntimeError(
                "cached teacher coordinate distillation requires a strictly "
                f"loaded representation {representation_key!r}"
            )
    _progressive_shared_head_weights(loss_cfg)
    if (
        isinstance(anchor_teacher_logit_cache, TeacherLogitCache)
        and anchor_teacher_logit_cache.audit.get("kind")
        == "fixed_offline_teacher_ensemble"
        and not bool(loss_cfg.get("cached_teacher_distill_correct_only", True))
    ):
        raise RuntimeError(
            "multi-teacher cached KD requires "
            "cached_teacher_distill_correct_only=true; teacher-wrong rows keep CE only"
        )
    if bool(
        loss_cfg.get(
            "rscd_pcgrad_enabled",
            train_cfg.get("rscd_pcgrad_enabled", False),
        )
    ):
        raise RuntimeError(
            "train.cached_teacher_fast_path cannot be used with PCGrad"
        )
    if int(train_cfg.get("save_step_checkpoint_every", 0) or 0) > 0:
        raise RuntimeError(
            "train.cached_teacher_fast_path currently requires "
            "save_step_checkpoint_every=0"
        )
    if int(
        train_cfg.get(
            "_resume_start_step",
            train_cfg.get("resume_start_step", 0),
        )
        or 0
    ) > 0:
        raise RuntimeError(
            "train.cached_teacher_fast_path requires an epoch-boundary start"
        )
    active_disallowed = {
        key: value
        for key, value in loss_cfg.items()
        if key not in _CACHED_TEACHER_FAST_LOSS_KEYS
        and not _inactive_loss_config_value(value)
    }
    if active_disallowed:
        raise RuntimeError(
            "train.cached_teacher_fast_path found active unsupported loss "
            f"settings: {sorted(active_disallowed)}"
        )


def _trace_lmcr_gradient_diagnostics(model: nn.Module) -> dict[str, float]:
    """Return small post-unscale gradient summaries for an LMCR backbone."""

    backbone = getattr(model, "backbone", None)
    steps = getattr(backbone, "local_moment_steps", None)
    transitions = getattr(backbone, "transitions", None)
    if not isinstance(steps, nn.ModuleList) or len(steps) != 4:
        return {}

    def grad_rms(parameter: torch.Tensor | None) -> float:
        gradient = getattr(parameter, "grad", None)
        if not isinstance(gradient, torch.Tensor):
            return 0.0
        return float(
            gradient.detach().float().square().mean().sqrt().cpu()
        )

    logs: dict[str, float] = {}
    for index, step in enumerate(steps):
        for name, parameter in (
            ("moment_projection", step.dispersion_projection.weight),
            ("context_projection", step.context_projection.weight),
            ("output_projection", step.output_projection.weight),
            ("gate", step.gate_logit),
        ):
            logs[f"trace_lmcr_grad_rms/{name}/stage{index + 1}"] = grad_rms(
                parameter
            )
        if isinstance(transitions, nn.ModuleList):
            transition_parameter = next(
                transitions[index].parameters(),
                None,
            )
            logs[f"trace_lmcr_grad_rms/main_transition/stage{index + 1}"] = (
                grad_rms(transition_parameter)
            )
    return logs


def _set_training_mode(
    model: nn.Module,
    cfg: dict[str, Any],
) -> None:
    """Enter train mode while keeping fully frozen submodules deterministic.

    Functional-recovery experiments compare a small trainable extension with
    logits produced by an eval-mode offline parent.  Calling ``model.train()``
    alone would reactivate Dropout and StochasticDepth inside the frozen
    carrier, even though none of their parameters can change.  Apply the same
    fail-closed mode contract to every training loop so the only stochastic or
    stateful training behaviour comes from modules that actually contain a
    trainable parameter.
    """

    model.train()
    if not bool(
        cfg.get("train", {}).get(
            "freeze_nontrainable_modules_eval",
            False,
        )
    ):
        return
    for module in model.modules():
        if not any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=True)
        ):
            module.eval()


def _train_one_epoch_cached_teacher_fast(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    *,
    anchor_teacher_logit_cache: dict[str, torch.Tensor],
    idx_to_class: dict[int, str],
    ema: ModelEMA | None,
) -> dict[str, float]:
    """Fast CE + cached-teacher loop for one deterministic training view."""

    _set_training_mode(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    progressive_weights = _progressive_shared_head_weights(loss_cfg)
    factor_weight = float(loss_cfg.get("factor_weight", 0.0))
    factor_aux_active = factor_weight > 0.0
    relation_weight = float(
        loss_cfg.get("cached_teacher_relation_distill_weight", 0.0)
    )
    coordinate_weight = float(
        loss_cfg.get("cached_teacher_coordinate_distill_weight", 0.0)
    )
    facet_joint_weight = float(
        loss_cfg.get("facet_joint_distill_weight", 0.0)
    )
    facc_node_aux_active = any(
        float(loss_cfg.get(key, 0.0)) > 0.0
        for key in (
            "facc_node_ce_weight_stage3",
            "facc_node_ce_weight_stage4",
        )
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = _amp_autocast_dtype(train_cfg, device) if use_amp else None
    accum = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    total_seen = 0
    total_loss = 0.0
    total_correct = 0
    total_steps = len(loader)
    log_every = int(train_cfg.get("log_every_steps", 80))
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    lmcr_gradient_sums: dict[str, float] = {}
    lmcr_gradient_updates = 0
    for step, batch in enumerate(
        tqdm(loader, desc="train-cache-kd-fast", leave=False, ascii=True),
        1,
    ):
        accumulation_window_size, optimizer_update_boundary = _accumulation_window(
            step,
            total_steps=total_steps,
            accumulation_steps=accum,
        )
        image = batch["image"].to(device, non_blocking=True)
        valid_mask = _batch_valid_mask(batch, device)
        observer_image = _batch_observer_image(batch, device)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            raw_output = _forward_surface_model(
                model,
                image,
                return_aux=(
                    progressive_weights is not None
                    or relation_weight > 0.0
                    or coordinate_weight > 0.0
                    or facet_joint_weight > 0.0
                    or facc_node_aux_active
                    or factor_aux_active
                ),
                valid_mask=valid_mask,
                observer_image=observer_image,
            )
            logits = (
                raw_output["logits"]
                if isinstance(raw_output, dict)
                else raw_output
            )
            teacher_logits, cache_logs = cached_teacher_logits_for_batch(
                anchor_teacher_logit_cache,
                batch["image_path"],
                device=device,
                # Keep the pinned teacher target bit-for-bit FP32.  Casting
                # cached logits to bf16 before correct-only selection changes
                # ties/argmax for a small but real subset of RSCD rows and
                # makes the selected supervision set depend on AMP.
                dtype=torch.float32,
                strict=True,
                cache_name="anchor",
                labels=label,
            )
            if teacher_logits is None:
                raise RuntimeError(
                    "strict cached-teacher fast path unexpectedly missed a batch"
                )
            relation_loss = logits.sum() * 0.0
            relation_logs: dict[str, float] = {}
            representation_cache_logs: dict[str, float] = {}
            if relation_weight > 0.0:
                if not isinstance(raw_output, Mapping):
                    raise RuntimeError(
                        "cached relation distillation requires a mapping model output"
                    )
                student_key = str(
                    loss_cfg.get(
                        "cached_teacher_relation_distill_student_key",
                        "backbone_embedding",
                    )
                ).strip()
                student_embedding = raw_output.get(student_key)
                if not isinstance(student_embedding, torch.Tensor):
                    raise RuntimeError(
                        "cached relation distillation model output lacks tensor "
                        f"{student_key!r}"
                    )
                representation_key = str(
                    loss_cfg.get(
                        "cached_teacher_relation_distill_representation_key",
                        "backbone_embedding",
                    )
                ).strip()
                teacher_embedding, representation_cache_logs = (
                    cached_teacher_representation_for_batch(
                        anchor_teacher_logit_cache,
                        batch["image_path"],
                        representation_key=representation_key,
                        device=device,
                        dtype=torch.float32,
                        strict=True,
                        cache_name="anchor",
                        labels=label,
                    )
                )
                if teacher_embedding is None:
                    raise RuntimeError(
                        "strict cached relation path unexpectedly missed a batch"
                    )
                relation_selection = _cached_teacher_selection_mask(
                    teacher_logits,
                    label,
                    loss_cfg,
                )
                relation_loss, relation_logs = (
                    _cached_teacher_relation_distillation_loss(
                        student_embedding,
                        teacher_embedding,
                        teacher_logits,
                        label,
                        loss_cfg,
                        selected_mask=relation_selection,
                    )
                )
            coordinate_loss = logits.sum() * 0.0
            coordinate_logs: dict[str, float] = {}
            if coordinate_weight > 0.0:
                if not isinstance(raw_output, Mapping):
                    raise RuntimeError(
                        "cached coordinate distillation requires a mapping "
                        "model output"
                    )
                student_key = str(
                    loss_cfg.get(
                        "cached_teacher_coordinate_distill_student_key",
                        "backbone_embedding",
                    )
                ).strip()
                student_embedding = raw_output.get(student_key)
                if not isinstance(student_embedding, torch.Tensor):
                    raise RuntimeError(
                        "cached coordinate distillation model output lacks "
                        f"tensor {student_key!r}"
                    )
                representation_key = str(
                    loss_cfg.get(
                        "cached_teacher_coordinate_distill_representation_key",
                        "backbone_embedding",
                    )
                ).strip()
                teacher_embedding, coordinate_cache_logs = (
                    cached_teacher_representation_for_batch(
                        anchor_teacher_logit_cache,
                        batch["image_path"],
                        representation_key=representation_key,
                        device=device,
                        dtype=torch.float32,
                        strict=True,
                        cache_name="anchor",
                        labels=label,
                    )
                )
                if teacher_embedding is None:
                    raise RuntimeError(
                        "strict cached coordinate path unexpectedly missed a batch"
                    )
                representation_cache_logs.update(coordinate_cache_logs)
                coordinate_selection = _cached_teacher_selection_mask(
                    teacher_logits,
                    label,
                    loss_cfg,
                )
                coordinate_loss, coordinate_logs = (
                    _cached_teacher_coordinate_distillation_loss(
                        student_embedding,
                        teacher_embedding,
                        teacher_logits,
                        label,
                        loss_cfg,
                        selected_mask=coordinate_selection,
                    )
                )
            if progressive_weights is None:
                main_loss = focus_weighted_cross_entropy(
                    logits,
                    label,
                    idx_to_class,
                    loss_cfg,
                )
                distill_loss, distill_logs = _cached_teacher_distillation_loss(
                    logits,
                    teacher_logits,
                    label,
                    loss_cfg,
                )
            else:
                if not isinstance(raw_output, Mapping):
                    raise RuntimeError(
                        "progressive cached-KD requires a mapping model output"
                    )
                logits, main_loss, distill_loss, distill_logs = (
                    _cached_teacher_progressive_objective(
                        raw_output,
                        teacher_logits,
                        label,
                        idx_to_class,
                        loss_cfg,
                    )
                )
            facet_joint_loss = logits.sum() * 0.0
            facet_joint_logs: dict[str, float] = {}
            if facet_joint_weight > 0.0:
                if not isinstance(raw_output, Mapping):
                    raise RuntimeError(
                        "FACET joint distillation requires a mapping model output"
                    )
                facet_joint_logits = raw_output.get("facet_joint_logits")
                if not isinstance(facet_joint_logits, torch.Tensor):
                    raise RuntimeError(
                        "FACET joint distillation model output lacks "
                        "facet_joint_logits"
                    )
                class_to_factor = getattr(
                    getattr(model, "spec", None),
                    "class_to_factor",
                    None,
                )
                if not isinstance(class_to_factor, torch.Tensor):
                    raise RuntimeError(
                        "FACET joint distillation requires model.spec.class_to_factor"
                    )
                facet_selection = _cached_teacher_selection_mask(
                    teacher_logits,
                    label,
                    loss_cfg,
                )
                facet_joint_loss, facet_joint_logs = (
                    _facet_joint_distillation_loss(
                        facet_joint_logits,
                        teacher_logits,
                        label,
                        class_to_factor,
                        loss_cfg,
                        selected_mask=facet_selection,
                    )
                )
            facc_node_loss = logits.sum() * 0.0
            facc_node_logs: dict[str, float] = {}
            if facc_node_aux_active:
                if not isinstance(raw_output, Mapping):
                    raise RuntimeError(
                        "FACC node auxiliary supervision requires a mapping model output"
                    )
                facc_node_loss, facc_node_logs = _facc_node_auxiliary_loss(
                    raw_output,
                    label,
                    loss_cfg,
                )
            factor_aux_loss = logits.sum() * 0.0
            factor_aux_logs: dict[str, float] = {}
            if factor_aux_active:
                if not isinstance(raw_output, Mapping):
                    raise RuntimeError(
                        "factor auxiliary supervision requires a mapping model output"
                    )
                spec = getattr(model, "spec", None)
                if spec is None:
                    raise RuntimeError(
                        "factor auxiliary supervision requires model.spec"
                    )
                factor_logits = raw_output.get("factor_logits")
                if not isinstance(factor_logits, Mapping):
                    raise RuntimeError(
                        "factor auxiliary supervision requires mapping "
                        "output['factor_logits']"
                    )
                axis_weights = loss_cfg.get("factor_axis_weights") or {}
                missing_factor_axes = [
                    axis
                    for axis in FACTOR_AXES
                    if float(axis_weights.get(axis, 1.0)) > 0.0
                    and not isinstance(factor_logits.get(axis), torch.Tensor)
                ]
                if missing_factor_axes:
                    raise RuntimeError(
                        "factor auxiliary supervision is missing active tensor "
                        f"heads: {missing_factor_axes}"
                    )
                # Admit only factor CE in this fast path.  The other C3
                # auxiliary objectives stay exactly disabled, and the pinned
                # teacher target above remains FP32 and student-independent.
                factor_aux_loss, factor_aux_logs = c3_total_aux_loss(
                    dict(raw_output),
                    label,
                    spec,
                    factor_weight=factor_weight,
                    factor_axis_weights=loss_cfg.get("factor_axis_weights"),
                    tournament_weight=0.0,
                    counterfactual_weight=0.0,
                    reliability_weight=0.0,
                    counterfactual_margin=1.0,
                    supervise_none=bool(loss_cfg.get("supervise_none", False)),
                )
            loss = (
                main_loss
                + distill_loss
                + relation_loss
                + coordinate_loss
                + facet_joint_loss
                + facc_node_loss
                + factor_aux_loss
            )
            backward = loss / float(accumulation_window_size)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(backward).backward()
        if optimizer_update_boundary:
            scaler.unscale_(optimizer)
            lmcr_gradient_logs = _trace_lmcr_gradient_diagnostics(model)
            if lmcr_gradient_logs:
                lmcr_gradient_updates += 1
                for key, value in lmcr_gradient_logs.items():
                    lmcr_gradient_sums[key] = (
                        lmcr_gradient_sums.get(key, 0.0) + float(value)
                    )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg.get("grad_clip_norm", 5.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int(
            (logits.argmax(dim=1) == label).sum().detach().cpu()
        )
        total_seen += batch_size
        diagnostic_count += 1
        for key, value in {
            **cache_logs,
            **representation_cache_logs,
            **distill_logs,
            **relation_logs,
            **coordinate_logs,
            **facet_joint_logs,
            **facc_node_logs,
            **factor_aux_logs,
        }.items():
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)
        if log_every > 0 and (step % log_every == 0 or step == total_steps):
            print(
                f"  train step {step}/{total_steps} "
                f"loss={total_loss/max(total_seen,1):.4f} "
                f"top1={total_correct/max(total_seen,1):.4f}"
            )
    metrics = {
        "loss": total_loss / max(total_seen, 1),
        "top1": total_correct / max(total_seen, 1),
        "cached_teacher_fast_path": 1.0,
    }
    if diagnostic_count:
        metrics.update(
            {
                key: value / diagnostic_count
                for key, value in diagnostic_sums.items()
            }
        )
    if lmcr_gradient_updates:
        metrics.update(
            {
                key: value / float(lmcr_gradient_updates)
                for key, value in lmcr_gradient_sums.items()
            }
        )
        metrics["trace_lmcr_gradient_updates"] = float(
            lmcr_gradient_updates
        )
    return metrics


_SELF_ANCHOR_FAST_LOSS_KEYS = {
    "label_smoothing",
    "focus_ce_extra_weight",
    "focus_ce_classes",
    "anchor_consistency_weight",
    "anchor_consistency_focus_weight",
    "anchor_consistency_temperature",
    "anchor_consistency_exempt_classes",
    "anchor_consistency_focus_low_margin_threshold",
    "anchor_consistency_protect_weight",
    "anchor_consistency_protect_confidence",
    "anchor_consistency_protect_margin",
    "anchor_no_flip_weight",
    "anchor_no_flip_nonfocus_only",
    "anchor_nonregression_weight",
    "anchor_nonregression_focus_weight",
    "anchor_nonregression_margin",
    "anchor_nonregression_confidence",
    "anchor_nonregression_teacher_margin",
    "anchor_nonregression_squared",
    "anchor_nonregression_focus_classes",
}


def _validate_self_anchor_fast_path(
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    teacher_model: nn.Module | None,
    expert_teacher_model: nn.Module | None,
    anchor_teacher_logit_cache: dict[str, torch.Tensor] | None,
    expert_teacher_logit_cache: dict[str, torch.Tensor] | None,
) -> None:
    """Fail closed for the CE + model-provided frozen-anchor specialization."""

    if str(getattr(model, "head_type", "")).strip().lower() != "linear":
        raise RuntimeError("train.self_anchor_fast_path requires a linear model head")
    if any(
        item is not None
        for item in (
            teacher_model,
            expert_teacher_model,
            anchor_teacher_logit_cache,
            expert_teacher_logit_cache,
        )
    ):
        raise RuntimeError(
            "train.self_anchor_fast_path requires the anchor logits exposed "
            "by the model and cannot be combined with external teachers"
        )
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    if bool(
        loss_cfg.get(
            "rscd_pcgrad_enabled",
            train_cfg.get("rscd_pcgrad_enabled", False),
        )
    ):
        raise RuntimeError("train.self_anchor_fast_path cannot be used with PCGrad")
    if int(train_cfg.get("save_step_checkpoint_every", 0) or 0) > 0:
        raise RuntimeError(
            "train.self_anchor_fast_path currently requires "
            "save_step_checkpoint_every=0"
        )
    if int(
        train_cfg.get(
            "_resume_start_step",
            train_cfg.get("resume_start_step", 0),
        )
        or 0
    ) > 0:
        raise RuntimeError(
            "train.self_anchor_fast_path requires an epoch-boundary start"
        )
    active_disallowed = {
        key: value
        for key, value in loss_cfg.items()
        if key not in _SELF_ANCHOR_FAST_LOSS_KEYS
        and not _inactive_loss_config_value(value)
    }
    if active_disallowed:
        raise RuntimeError(
            "train.self_anchor_fast_path found active unsupported loss "
            f"settings: {sorted(active_disallowed)}"
        )
    anchor_weight = sum(
        max(float(loss_cfg.get(key, 0.0) or 0.0), 0.0)
        for key in (
            "anchor_consistency_weight",
            "anchor_consistency_focus_weight",
            "anchor_consistency_protect_weight",
            "anchor_no_flip_weight",
            "anchor_nonregression_weight",
            "anchor_nonregression_focus_weight",
        )
    )
    if anchor_weight <= 0.0:
        raise RuntimeError(
            "train.self_anchor_fast_path requires at least one active anchor "
            "loss; use linear_ce_fast_path for plain CE"
        )


def _train_one_epoch_self_anchor_fast(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    *,
    idx_to_class: dict[int, str],
    ema: ModelEMA | None,
) -> dict[str, float]:
    """Exact CE + internal frozen-anchor training specialization."""

    _set_training_mode(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = _amp_autocast_dtype(train_cfg, device) if use_amp else None
    accum = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    total_seen = 0
    total_loss = 0.0
    total_correct = 0
    total_steps = len(loader)
    log_every = int(train_cfg.get("log_every_steps", 80))
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    for step, batch in enumerate(
        tqdm(loader, desc="train-self-anchor", leave=False, ascii=True),
        1,
    ):
        accumulation_window_size, optimizer_update_boundary = _accumulation_window(
            step,
            total_steps=total_steps,
            accumulation_steps=accum,
        )
        image = batch["image"].to(device, non_blocking=True)
        valid_mask = _batch_valid_mask(batch, device)
        observer_image = _batch_observer_image(batch, device)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            out = _forward_surface_model(
                model,
                image,
                return_aux=True,
                valid_mask=valid_mask,
                observer_image=observer_image,
            )
            if not isinstance(out, dict):
                raise RuntimeError(
                    "self-anchor model must return an auxiliary dictionary"
                )
            logits = out.get("logits")
            anchor_logits = out.get("anchor_logits")
            if not isinstance(logits, torch.Tensor) or not isinstance(
                anchor_logits, torch.Tensor
            ):
                raise RuntimeError(
                    "self-anchor model output must contain tensor logits and "
                    "anchor_logits"
                )
            if tuple(anchor_logits.shape) != tuple(logits.shape):
                raise RuntimeError(
                    "self-anchor logits shape mismatch: "
                    f"{tuple(anchor_logits.shape)} versus {tuple(logits.shape)}"
                )
            main_loss = focus_weighted_cross_entropy(
                logits,
                label,
                idx_to_class,
                loss_cfg,
            )
            anchor_loss, anchor_logs = anchor_consistency_loss(
                logits,
                anchor_logits.detach(),
                label,
                idx_to_class,
                loss_cfg,
            )
            nonregression_loss, nonregression_logs = (
                anchor_nonregression_barrier_loss(
                    logits,
                    anchor_logits.detach(),
                    label,
                    idx_to_class,
                    loss_cfg,
                )
            )
            loss = main_loss + anchor_loss + nonregression_loss
            backward = loss / float(accumulation_window_size)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(backward).backward()
        if optimizer_update_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg.get("grad_clip_norm", 5.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int(
            (logits.argmax(dim=1) == label).sum().detach().cpu()
        )
        total_seen += batch_size
        diagnostic_count += 1
        for key, value in {**anchor_logs, **nonregression_logs}.items():
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)
        if log_every > 0 and (step % log_every == 0 or step == total_steps):
            print(
                f"  train step {step}/{total_steps} "
                f"loss={total_loss/max(total_seen,1):.4f} "
                f"top1={total_correct/max(total_seen,1):.4f}"
            )
    metrics = {
        "loss": total_loss / max(total_seen, 1),
        "top1": total_correct / max(total_seen, 1),
        "self_anchor_fast_path": 1.0,
    }
    if diagnostic_count:
        metrics.update(
            {
                key: value / diagnostic_count
                for key, value in diagnostic_sums.items()
            }
        )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict[str, Any],
    scaler: torch.amp.GradScaler,
    *,
    teacher_model: C3FaRNetSurfaceClassifier | None = None,
    expert_teacher_model: C3FaRNetSurfaceClassifier | None = None,
    anchor_teacher_logit_cache: dict[str, torch.Tensor] | None = None,
    expert_teacher_logit_cache: dict[str, torch.Tensor] | None = None,
    teacher_cache_strict: bool = False,
    idx_to_class: dict[int, str] | None = None,
    out_dir: Path | None = None,
    epoch: int = 1,
    class_to_idx: dict[str, int] | None = None,
    ema: ModelEMA | None = None,
    run_provenance: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, float]:
    """完成一个 epoch，并按配置选择唯一训练路径。

    当前正式配置把 ``linear_ce_fast_path`` 设为 false，原因是 Balanced
    Softmax 必须经过通用 CE 路径。其余 fast path 是历史消融兼容代码；三个
    fast path 互斥。函数返回训练 loss、top1 及审计指标。
    """
    fast_path_flags = {
        "linear_ce_fast_path": bool(
            cfg.get("train", {}).get("linear_ce_fast_path", False)
        ),
        "self_anchor_fast_path": bool(
            cfg.get("train", {}).get("self_anchor_fast_path", False)
        ),
        "cached_teacher_fast_path": bool(
            cfg.get("train", {}).get("cached_teacher_fast_path", False)
        ),
    }
    active_fast_paths = [name for name, active in fast_path_flags.items() if active]
    if len(active_fast_paths) > 1:
        raise RuntimeError(
            "training fast paths are mutually exclusive: "
            f"{active_fast_paths}"
        )
    if fast_path_flags["linear_ce_fast_path"]:
        _validate_linear_ce_fast_path(
            model,
            cfg,
            teacher_model=teacher_model,
            expert_teacher_model=expert_teacher_model,
            anchor_teacher_logit_cache=anchor_teacher_logit_cache,
            expert_teacher_logit_cache=expert_teacher_logit_cache,
            out_dir=out_dir,
        )
        return _train_one_epoch_linear_ce_fast(
            model,
            loader,
            optimizer,
            device,
            cfg,
            scaler,
            idx_to_class=idx_to_class or {},
            ema=ema,
            out_dir=out_dir,
            epoch=epoch,
            class_to_idx=class_to_idx,
            run_provenance=run_provenance,
        )
    if fast_path_flags["cached_teacher_fast_path"]:
        _validate_cached_teacher_fast_path(
            model,
            cfg,
            teacher_model=teacher_model,
            expert_teacher_model=expert_teacher_model,
            anchor_teacher_logit_cache=anchor_teacher_logit_cache,
            expert_teacher_logit_cache=expert_teacher_logit_cache,
            teacher_cache_strict=teacher_cache_strict,
        )
        assert anchor_teacher_logit_cache is not None
        return _train_one_epoch_cached_teacher_fast(
            model,
            loader,
            optimizer,
            device,
            cfg,
            scaler,
            anchor_teacher_logit_cache=anchor_teacher_logit_cache,
            idx_to_class=idx_to_class or {},
            ema=ema,
        )
    if fast_path_flags["self_anchor_fast_path"]:
        _validate_self_anchor_fast_path(
            model,
            cfg,
            teacher_model=teacher_model,
            expert_teacher_model=expert_teacher_model,
            anchor_teacher_logit_cache=anchor_teacher_logit_cache,
            expert_teacher_logit_cache=expert_teacher_logit_cache,
        )
        return _train_one_epoch_self_anchor_fast(
            model,
            loader,
            optimizer,
            device,
            cfg,
            scaler,
            idx_to_class=idx_to_class or {},
            ema=ema,
        )
    if any(
        float(cfg.get("loss", {}).get(key, 0.0)) > 0.0
        for key in (
            "cached_teacher_relation_distill_weight",
            "cached_teacher_coordinate_distill_weight",
        )
    ):
        raise RuntimeError(
            "cached teacher representation distillation is currently defined "
            "only for train.cached_teacher_fast_path"
        )
    _set_training_mode(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    pcgrad_enabled = bool(loss_cfg.get("rscd_pcgrad_enabled", train_cfg.get("rscd_pcgrad_enabled", False)))
    grouped_pcgrad_enabled = bool(loss_cfg.get("rscd_pcgrad_grouped_protect_enabled", False))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda" and not pcgrad_enabled
    amp_dtype = _amp_autocast_dtype(train_cfg, device) if use_amp else None
    accum = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    pareto_selected_edge_rules = prepare_pareto_selected_edge_rules(loss_cfg)
    resume_start_step = int(train_cfg.get("_resume_start_step", train_cfg.get("resume_start_step", 0)) or 0)
    resume_partial = (
        train_cfg.get("_resume_train_partial", {})
        if resume_start_step > 0
        else {}
    )
    if not isinstance(resume_partial, dict):
        resume_partial = {}
    total_seen = max(int(resume_partial.get("seen", 0) or 0), 0)
    total_loss = float(resume_partial.get("loss", 0.0) or 0.0) * total_seen
    if "correct" in resume_partial:
        total_correct = max(int(resume_partial.get("correct", 0) or 0), 0)
    else:
        total_correct = int(
            round(float(resume_partial.get("top1", 0.0) or 0.0) * total_seen)
        )
    log_every = int(train_cfg.get("log_every_steps", 80))
    raw_aux_log_sum = resume_partial.get("aux_log_sum", {})
    aux_log_sum: dict[str, float] = (
        {str(key): float(value) for key, value in raw_aux_log_sum.items()}
        if isinstance(raw_aux_log_sum, dict)
        else {}
    )
    aux_log_count = max(int(resume_partial.get("aux_log_count", 0) or 0), 0)
    total_steps = resume_start_step + len(loader)
    step_checkpoint_every = int(train_cfg.get("save_step_checkpoint_every", 0) or 0)
    last_step_checkpoint_step = resume_start_step
    # 恢复时 loader 已从断点后的数据位置开始，local_step 再加 resume_start_step
    # 即得到本 epoch 的绝对 step（当前首次应从 2501 继续）。
    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} train",
        total=total_steps,
        initial=resume_start_step,
        unit="batch",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    )
    for local_step, batch in enumerate(progress_bar, 1):
        step = resume_start_step + local_step
        accumulation_window_size, optimizer_update_boundary = _accumulation_window(
            step,
            total_steps=total_steps,
            accumulation_steps=accum,
        )
        completed_updates_before_step = (
            _completed_optimizer_updates_before_step(
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                accumulation_steps=accum,
            )
        )
        _set_optimizer_update_curricula(
            model,
            completed_updates_before_step,
        )
        # CPU DataLoader → GPU；non_blocking 与 pin_memory 配合只影响速度。
        image = batch["image"].to(device, non_blocking=True)
        valid_mask = _batch_valid_mask(batch, device)
        observer_image = _batch_observer_image(batch, device)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            # 前向返回 raw 27 类 logits 和模型内部审计量；BF16 只包围允许的算子。
            out = _forward_surface_model(
                model,
                image,
                return_aux=True,
                valid_mask=valid_mask,
                observer_image=observer_image,
            )
            logits = out["logits"]
            ce_logs: dict[str, float] = {}
            if float(loss_cfg.get("feature_mechanism_ce_extra_weight", 0.0)) > 0.0:
                main_loss, ce_logs = mechanism_feature_weighted_cross_entropy(
                    logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                    out,
                )
            else:
                # 正式 seed97 在这里计算 Balanced Softmax CE。
                main_loss = focus_weighted_cross_entropy(logits, label, idx_to_class or {}, loss_cfg)
            aux_loss, aux_logs = c3_total_aux_loss(
                out,
                label,
                model.spec,
                factor_weight=float(loss_cfg.get("factor_weight", 0.3)),
                factor_axis_weights=loss_cfg.get("factor_axis_weights", {"friction": 1.0, "material": 1.0, "roughness": 1.0}),
                tournament_weight=float(loss_cfg.get("tournament_weight", 0.1)),
                counterfactual_weight=float(loss_cfg.get("counterfactual_weight", 0.05)),
                reliability_weight=float(loss_cfg.get("reliability_weight", 0.05)),
                counterfactual_margin=float(loss_cfg.get("counterfactual_margin", 1.0)),
                supervise_none=bool(loss_cfg.get("supervise_none", False)),
            )
            binary_loss, binary_logs = hardpair_binary_tournament_loss(out, label, model.spec, loss_cfg)
            value_pair_loss, value_pair_logs = hardpair_value_adapter_pairwise_loss(out, label, model.spec, loss_cfg)
            feature_value_pair_loss, feature_value_pair_logs = feature_value_boundary_pairwise_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            opponent_feature_pair_loss, opponent_feature_pair_logs = water_concrete_opponent_feature_pairwise_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            rough_order_loss, rough_order_logs = value_guided_roughness_order_loss(
                out,
                label,
                model.spec,
                loss_cfg,
                idx_to_class or {},
            )
            protected_tristate_loss, protected_tristate_logs = protected_tristate_roughness_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            roughness_fiber_ce_loss, roughness_fiber_ce_logs = (
                roughness_fiber_conditional_ce_loss(
                    out,
                    label,
                    model.spec,
                    loss_cfg,
                )
            )
            value_margin_loss, value_margin_logs = value_guided_hardpair_margin_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            pair_value_selective_loss, pair_value_selective_logs = pair_value_selective_margin_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            marginal_consistency_loss, marginal_consistency_logs = factor_marginal_consistency_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            dry_ordinal_loss, dry_ordinal_logs = dry_concrete_bidirectional_ordinal_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            family_router_loss, family_router_logs = family_mechanism_router_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            selected_edge_loss, selected_edge_logs = pareto_selected_edge_margin_loss(
                logits,
                label,
                idx_to_class or {},
                pareto_selected_edge_rules,
                loss_cfg,
            )
            graph_metric_loss, graph_metric_logs = factor_graph_metric_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            ceta_loss, ceta_logs = ceta_calibration_loss(
                out,
                label,
                model.spec,
                wet_weight=float(loss_cfg.get("ceta_wet_calibration_weight", 0.0)),
                dry_expectation_weight=float(
                    loss_cfg.get("ceta_dry_expectation_weight", 0.0)
                ),
                expectation_beta=float(
                    loss_cfg.get("ceta_expectation_smooth_l1_beta", 0.25)
                ),
            )
            teacher_logits: torch.Tensor | None = None
            anchor_loss = logits.new_zeros(())
            nonregression_loss = logits.new_zeros(())
            pareto_safe_loss = logits.new_zeros(())
            classwise_pareto_loss = logits.new_zeros(())
            dual_teacher_loss = logits.new_zeros(())
            s7_drel_gate_loss = logits.new_zeros(())
            gate_loss = logits.new_zeros(())
            margin_loss = logits.new_zeros(())
            feature_distill_loss = logits.new_zeros(())
            backbone_stage_distill_loss = logits.new_zeros(())
            anchor_logs: dict[str, float] = {}
            nonregression_logs: dict[str, float] = {}
            pareto_safe_logs: dict[str, float] = {}
            classwise_pareto_logs: dict[str, float] = {}
            dual_teacher_logs: dict[str, float] = {}
            s7_drel_gate_logs: dict[str, float] = {}
            gate_logs: dict[str, float] = {}
            margin_logs: dict[str, float] = {}
            feature_distill_logs: dict[str, float] = {}
            backbone_stage_distill_logs: dict[str, float] = {}
            cache_logs: dict[str, float] = {}
            expert_teacher_logits: torch.Tensor | None = None
            teacher_aux_out: dict[str, Any] | None = None
            teacher_logits, anchor_cache_logs = cached_teacher_logits_for_batch(
                anchor_teacher_logit_cache,
                batch["image_path"],
                device=device,
                dtype=logits.dtype,
                strict=teacher_cache_strict,
                cache_name="anchor",
                labels=label,
            )
            cache_logs.update(anchor_cache_logs)
            if teacher_logits is None and teacher_model is not None:
                with torch.no_grad():
                    if float(loss_cfg.get("teacher_feature_distill_weight", 0.0)) > 0.0:
                        teacher_raw = _forward_surface_model(
                            teacher_model,
                            image,
                            return_aux=True,
                            valid_mask=valid_mask,
                        )
                        if isinstance(teacher_raw, dict):
                            teacher_aux_out = teacher_raw
                            teacher_logits = teacher_raw["logits"]
                        else:
                            teacher_logits = teacher_raw
                    else:
                        teacher_logits = _forward_surface_model(
                            teacher_model,
                            image,
                            return_aux=False,
                            valid_mask=valid_mask,
                        )
            if teacher_logits is None and isinstance(out, dict):
                # A zero-start residual classifier may expose the frozen
                # carrier prediction it already computed in the same forward.
                # Reusing it avoids a second public-parent pass or a test-set
                # cache while keeping every anchor objective generic.
                internal_anchor = out.get("anchor_logits")
                if isinstance(internal_anchor, torch.Tensor):
                    if tuple(internal_anchor.shape) != tuple(logits.shape):
                        raise ValueError(
                            "model-provided anchor_logits must match logits: "
                            f"anchor={tuple(internal_anchor.shape)} "
                            f"student={tuple(logits.shape)}"
                        )
                    teacher_logits = internal_anchor.detach()
                    cache_logs["anchor_self_model"] = 1.0
            if teacher_logits is not None:
                anchor_loss, anchor_logs = anchor_consistency_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                )
                nonregression_loss, nonregression_logs = anchor_nonregression_barrier_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                )
                pareto_safe_loss, pareto_safe_logs = pareto_safe_distillation_loss(
                    logits,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    model.spec,
                    loss_cfg,
                )
                classwise_pareto_loss, classwise_pareto_logs = classwise_pareto_groupdro_loss(
                    out,
                    teacher_logits,
                    label,
                    idx_to_class or {},
                    model.spec,
                    loss_cfg,
                )
                gate_loss, gate_logs = anchor_error_gate_loss(out, teacher_logits, label, model.spec, loss_cfg)
                margin_loss, margin_logs = hardpair_margin_directed_loss(out, teacher_logits, label, model.spec, loss_cfg)
            feature_distill_loss, feature_distill_logs = teacher_feature_distillation_loss(
                out,
                teacher_aux_out,
                loss_cfg,
            )
            (
                backbone_stage_distill_loss,
                backbone_stage_distill_logs,
            ) = backbone_stage_replacement_distillation_loss(
                model,
                logits,
                loss_cfg,
            )
            arcq_consistency_loss, arcq_consistency_logs = arcq_composition_consistency_loss(
                model,
                image,
                out,
                loss_cfg,
            )
            arcq_coral_loss, arcq_coral_logs = roughness_coral_aux_loss(
                out,
                label,
                model.spec,
                loss_cfg,
            )
            proxy_compactness_loss, proxy_compactness_logs = (
                classifier_proxy_compactness_loss(
                    model,
                    out,
                    label,
                    loss_cfg,
                )
            )
            expert_teacher_logits, expert_cache_logs = cached_teacher_logits_for_batch(
                expert_teacher_logit_cache,
                batch["image_path"],
                device=device,
                dtype=logits.dtype,
                strict=teacher_cache_strict,
                cache_name="expert",
                labels=label,
            )
            cache_logs.update(expert_cache_logs)
            if expert_teacher_logits is None and expert_teacher_model is not None:
                with torch.no_grad():
                    expert_teacher_logits = _forward_surface_model(
                        expert_teacher_model,
                        image,
                        return_aux=False,
                        valid_mask=valid_mask,
                    )
            if expert_teacher_logits is None and isinstance(out, dict):
                # Dual-encoder students may already contain a frozen specialist.
                # Reuse that exact prediction instead of running or caching a
                # second copy of the expert during training.
                internal_expert = out.get("expert_logits")
                if isinstance(internal_expert, torch.Tensor):
                    if tuple(internal_expert.shape) != tuple(logits.shape):
                        raise ValueError(
                            "model-provided expert_logits must match logits: "
                            f"expert={tuple(internal_expert.shape)} "
                            f"student={tuple(logits.shape)}"
                        )
                    expert_teacher_logits = internal_expert.detach()
                    cache_logs["expert_self_model"] = 1.0
            if teacher_logits is not None or expert_teacher_logits is not None:
                dual_teacher_loss, dual_teacher_logs = dual_teacher_noharm_loss(
                    logits,
                    teacher_logits,
                    expert_teacher_logits,
                    label,
                    idx_to_class or {},
                    loss_cfg,
                )
            s7_drel_gate_loss, s7_drel_gate_logs = s7_drel_rescue_gate_loss(
                out,
                teacher_logits,
                expert_teacher_logits,
                label,
                loss_cfg,
            )
            pcgrad_focus_loss: torch.Tensor | None = None
            pcgrad_protect_loss: torch.Tensor | None = None
            pcgrad_protect_group_losses: list[tuple[str, torch.Tensor]] = []
            pcgrad_logs: dict[str, float] = {}
            pcgrad_focus_gradient_weight = float(loss_cfg.get("rscd_pcgrad_focus_weight", 0.0))
            if pcgrad_enabled:
                if grouped_pcgrad_enabled:
                    pcgrad_focus_loss, pcgrad_protect_group_losses, pcgrad_logs = rscd_focus_grouped_protect_objectives(
                        out,
                        teacher_logits,
                        label,
                        idx_to_class or {},
                        model.spec,
                        loss_cfg,
                    )
                    if pcgrad_protect_group_losses:
                        pcgrad_protect_loss = torch.stack([item[1] for item in pcgrad_protect_group_losses]).mean()
                else:
                    pcgrad_focus_loss, pcgrad_protect_loss, pcgrad_logs = rscd_focus_protect_objectives(
                        logits,
                        teacher_logits,
                        label,
                        idx_to_class or {},
                        loss_cfg,
                    )
            agem_memory_grads: list[torch.Tensor | None] | None = None
            if bool(loss_cfg.get("rscd_agem_total_projection_enabled", False)) and (
                pcgrad_protect_loss is not None or pcgrad_protect_group_losses
            ):
                agem_memory_grads = rscd_collect_protect_memory_gradient(
                    trainable_params,
                    pcgrad_protect_loss,
                    pcgrad_protect_group_losses,
                )
            main_loss_for_total = main_loss
            if bool(loss_cfg.get("rscd_pcgrad_decompose_main_ce", False)) and pcgrad_protect_loss is not None:
                protect_loss_weight = float(loss_cfg.get("rscd_pcgrad_protect_loss_weight", 1.0))
                if bool(loss_cfg.get("rscd_pcgrad_preserve_batch_ce_scale", True)):
                    protect_loss_weight *= float(pcgrad_logs.get("rscd_pcgrad_protect_count", 0.0)) / max(float(label.numel()), 1.0)
                    pcgrad_focus_gradient_weight *= float(pcgrad_logs.get("rscd_pcgrad_focus_count", 0.0)) / max(
                        float(label.numel()), 1.0
                    )
                main_loss_for_total = protect_loss_weight * pcgrad_protect_loss.to(dtype=logits.dtype)
                pcgrad_logs["loss_main_decomposed_protect_ce"] = float(main_loss_for_total.detach().cpu())
                pcgrad_logs["rscd_pcgrad_effective_focus_weight"] = float(pcgrad_focus_gradient_weight)
            loss = (
                main_loss_for_total
                + aux_loss
                + binary_loss
                + value_pair_loss
                + feature_value_pair_loss
                + opponent_feature_pair_loss
                + rough_order_loss
                + protected_tristate_loss
                + roughness_fiber_ce_loss
                + value_margin_loss
                + pair_value_selective_loss
                + marginal_consistency_loss
                + dry_ordinal_loss
                + family_router_loss
                + selected_edge_loss
                + graph_metric_loss
                + ceta_loss
                + anchor_loss
                + nonregression_loss
                + pareto_safe_loss
                + classwise_pareto_loss
                + dual_teacher_loss
                + s7_drel_gate_loss
                + gate_loss
                + margin_loss
                + feature_distill_loss
                + backbone_stage_distill_loss
                + arcq_consistency_loss
                + arcq_coral_loss
                + proxy_compactness_loss
            )
            # 梯度累积 2 步：每个 micro-batch 的 loss 除以窗口大小，再累积梯度。
            backward = loss / float(accumulation_window_size)
        if not bool(torch.isfinite(loss.detach())):
            optimizer.zero_grad(set_to_none=True)
            continue
        if pcgrad_enabled:
            pcgrad_adjusted_grads: list[torch.Tensor | None] | None = None
            if pcgrad_focus_loss is not None and pcgrad_protect_loss is not None:
                if grouped_pcgrad_enabled and pcgrad_protect_group_losses:
                    pcgrad_adjusted_grads, pcgrad_surgery_logs = rscd_focus_grouped_protect_gradient_surgery(
                        trainable_params,
                        pcgrad_focus_loss,
                        pcgrad_protect_group_losses,
                        focus_weight=float(pcgrad_focus_gradient_weight),
                        accum=accumulation_window_size,
                    )
                else:
                    pcgrad_adjusted_grads, pcgrad_surgery_logs = rscd_focus_protect_gradient_surgery(
                        trainable_params,
                        pcgrad_focus_loss,
                        pcgrad_protect_loss,
                        focus_weight=float(pcgrad_focus_gradient_weight),
                        accum=accumulation_window_size,
                    )
                pcgrad_logs.update(pcgrad_surgery_logs)
            backward.backward()
            if pcgrad_adjusted_grads is not None:
                with torch.no_grad():
                    for param, grad in zip(trainable_params, pcgrad_adjusted_grads):
                        if grad is None:
                            continue
                        if param.grad is None:
                            param.grad = grad.detach().clone()
                        else:
                            param.grad.add_(grad.to(device=param.grad.device, dtype=param.grad.dtype))
            if agem_memory_grads is not None:
                pcgrad_logs.update(rscd_project_total_gradient_against_memory(trainable_params, agem_memory_grads))
            if optimizer_update_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 5.0)))
                optimizer.step()
                _set_optimizer_update_curricula(
                    model,
                    completed_updates_before_step + 1,
                )
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
        else:
            # BF16 时 GradScaler 实际关闭，但统一接口保证 FP16 历史配置仍兼容。
            scaler.scale(backward).backward()
            if optimizer_update_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip_norm", 5.0)))
                # 只有到梯度累积边界才真正更新一次参数。
                scaler.step(optimizer)
                scaler.update()
                _set_optimizer_update_curricula(
                    model,
                    completed_updates_before_step + 1,
                )
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
        batch_size = int(label.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == label).sum().detach().cpu())
        total_seen += batch_size
        aux_log_count += 1
        for key, value in aux_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in ce_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in binary_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in value_pair_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in feature_value_pair_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in opponent_feature_pair_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in rough_order_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in protected_tristate_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in roughness_fiber_ce_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in value_margin_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in pair_value_selective_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in marginal_consistency_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in dry_ordinal_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in family_router_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in selected_edge_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in graph_metric_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in ceta_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in anchor_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in nonregression_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in pareto_safe_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in classwise_pareto_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in dual_teacher_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in s7_drel_gate_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in cache_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in gate_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in margin_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in feature_distill_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in backbone_stage_distill_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in arcq_consistency_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in arcq_coral_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in proxy_compactness_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        for key, value in pcgrad_logs.items():
            aux_log_sum[key] = aux_log_sum.get(key, 0.0) + float(value)
        checkpoint_due = (
            step_checkpoint_every > 0
            and out_dir is not None
            and optimizer_update_boundary
            and (
                step - last_step_checkpoint_step >= step_checkpoint_every
                or local_step == len(loader)
            )
        )
        if checkpoint_due:
            # 只在 optimizer update 边界保存，确保没有“尚未序列化的半窗梯度”。
            step_state = _training_step_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader=loader,
                cfg=cfg,
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                class_to_idx=class_to_idx,
                run_provenance=run_provenance,
                train_partial={
                    "loss": total_loss / max(total_seen, 1),
                    "top1": total_correct / max(total_seen, 1),
                    "correct": int(total_correct),
                    "seen": int(total_seen),
                    "aux_log_sum": dict(aux_log_sum),
                    "aux_log_count": int(aux_log_count),
                },
                ema=ema,
            )
            _atomic_torch_save(step_state, out_dir / "last_step_checkpoint.pth")
            last_step_checkpoint_step = int(step)
            print(f"  saved step checkpoint: {out_dir / 'last_step_checkpoint.pth'} step={step}/{total_steps}")
        if log_every > 0 and (step % log_every == 0 or local_step == len(loader)):
            running_loss = total_loss / max(total_seen, 1)
            running_top1 = total_correct / max(total_seen, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            progress_bar.set_postfix(
                loss=f"{running_loss:.4f}",
                top1=f"{100.0 * running_top1:.2f}%",
                lr=f"{current_lr:.2e}",
                refresh=True,
            )
            print(f"  train step {step}/{total_steps} loss={running_loss:.4f} top1={running_top1:.4f}")
            if progress_callback is not None:
                progress_callback(
                    epoch=epoch,
                    step=step,
                    total_steps=total_steps,
                    loss=running_loss,
                    top1=running_top1,
                    learning_rate=current_lr,
                )
    logs = {key: value / max(aux_log_count, 1) for key, value in aux_log_sum.items()}
    logs.update({"loss": total_loss / max(total_seen, 1), "top1": total_correct / max(total_seen, 1)})
    logs.update(_optimizer_update_curriculum_logs(model))
    return logs


def _pair_classification_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    *,
    left_idx: int,
    right_idx: int,
) -> dict[str, float | int]:
    """Separate direct pair discrimination from third-class errors.

    The historical ``pair_accuracy`` evaluates the final 27-way prediction on
    samples whose truth belongs to a selected pair.  It can improve merely
    because a sample stops going to a third class, even when the two members
    remain directly confused.  This helper additionally compares only the two
    selected logits and reports directional 27-way confusions explicitly.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError("pair diagnostics require matching one-dimensional labels")
    if logits.ndim != 2 or logits.shape[0] != y_true.shape[0]:
        raise ValueError("pair diagnostics logits must be NxK and match labels")
    left_idx, right_idx = int(left_idx), int(right_idx)
    if min(left_idx, right_idx) < 0:
        return {
            "pair_samples": 0,
            "restricted_27way_accuracy": 0.0,
            "binary_logit_accuracy": 0.0,
            "left_to_right_27way": 0,
            "right_to_left_27way": 0,
            "direct_27way_confusions": 0,
            "third_class_errors": 0,
        }
    if max(left_idx, right_idx) >= logits.shape[1]:
        raise ValueError("pair diagnostic class index is outside the logit matrix")
    mask = (y_true == left_idx) | (y_true == right_idx)
    pair_samples = int(mask.sum())
    if pair_samples == 0:
        return {
            "pair_samples": 0,
            "restricted_27way_accuracy": 0.0,
            "binary_logit_accuracy": 0.0,
            "left_to_right_27way": 0,
            "right_to_left_27way": 0,
            "direct_27way_confusions": 0,
            "third_class_errors": 0,
        }
    selected = logits[mask][:, [left_idx, right_idx]]
    binary_choice = selected.argmax(axis=1)
    binary_pred = np.where(binary_choice == 0, left_idx, right_idx)
    truth = y_true[mask]
    prediction = y_pred[mask]
    left_to_right = int(((truth == left_idx) & (prediction == right_idx)).sum())
    right_to_left = int(((truth == right_idx) & (prediction == left_idx)).sum())
    is_pair_prediction = (prediction == left_idx) | (prediction == right_idx)
    return {
        "pair_samples": pair_samples,
        "restricted_27way_accuracy": float((truth == prediction).mean()),
        "binary_logit_accuracy": float((truth == binary_pred).mean()),
        "left_to_right_27way": left_to_right,
        "right_to_left_27way": right_to_left,
        "direct_27way_confusions": int(left_to_right + right_to_left),
        "third_class_errors": int((~is_pair_prediction).sum()),
    }


def _ordered_evaluation_hashes(
    image_paths: Sequence[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
) -> dict[str, str]:
    """Hash the exact ordered FP32 evaluation payload.

    Metric equality is too weak for paired mechanism screens: two models can
    have identical Top-1/F1 while their logits already differ.  These hashes
    bind the canonical path order, labels, predictions, and the complete
    little-endian FP32 logit matrix without persisting the large matrix itself.
    The combined digest is length-delimited so path concatenation cannot create
    an ambiguous payload.
    """

    canonical_paths = [
        str(path).strip().replace("\\", "/").casefold() for path in image_paths
    ]
    labels_i64 = np.ascontiguousarray(labels, dtype=np.dtype("<i8"))
    predictions_i64 = np.ascontiguousarray(
        predictions,
        dtype=np.dtype("<i8"),
    )
    logits_f32 = np.ascontiguousarray(logits, dtype=np.dtype("<f4"))

    path_digest = hashlib.sha256()
    combined_digest = hashlib.sha256()
    for path in canonical_paths:
        encoded = path.encode("utf-8")
        framed = len(encoded).to_bytes(8, byteorder="little", signed=False) + encoded
        path_digest.update(framed)
        combined_digest.update(framed)

    labels_payload = labels_i64.tobytes(order="C")
    predictions_payload = predictions_i64.tobytes(order="C")
    logits_payload = logits_f32.tobytes(order="C")
    for payload in (labels_payload, predictions_payload, logits_payload):
        combined_digest.update(
            len(payload).to_bytes(8, byteorder="little", signed=False)
        )
        combined_digest.update(payload)

    return {
        "evaluation_path_order_sha256": path_digest.hexdigest(),
        "evaluation_labels_int64_sha256": hashlib.sha256(labels_payload).hexdigest(),
        "evaluation_predictions_int64_sha256": hashlib.sha256(
            predictions_payload
        ).hexdigest(),
        "evaluation_logits_fp32_sha256": hashlib.sha256(logits_payload).hexdigest(),
        "evaluation_payload_sha256": combined_digest.hexdigest(),
    }


_TOURNAMENT_DIAGNOSTIC_GROUPS = (
    "hard",
    "friction",
    "material",
    "roughness",
)


def _new_tournament_diagnostic_accumulator() -> dict[str, dict[str, float]]:
    return {
        name: {"correct": 0.0, "count": 0.0}
        for name in _TOURNAMENT_DIAGNOSTIC_GROUPS
    }


def _accumulate_tournament_diagnostics(
    accumulator: dict[str, dict[str, float]],
    batch_logs: Mapping[str, float],
) -> None:
    """Merge pair diagnostics without making them depend on batch partitioning."""

    for name in _TOURNAMENT_DIAGNOSTIC_GROUPS:
        prefix = "hard_pair" if name == "hard" else f"{name}_pair"
        count = float(batch_logs[f"{prefix}_count"])
        accuracy = float(batch_logs[f"{prefix}_acc"])
        if not math.isfinite(count) or count < 0.0:
            raise RuntimeError(
                f"{prefix}_count must be finite and non-negative, got {count}"
            )
        if not math.isfinite(accuracy) or not (0.0 <= accuracy <= 1.0):
            raise RuntimeError(
                f"{prefix}_acc must be finite and in [0,1], got {accuracy}"
            )
        accumulator[name]["count"] += count
        accumulator[name]["correct"] += accuracy * count


def _finalize_tournament_diagnostics(
    accumulator: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name in _TOURNAMENT_DIAGNOSTIC_GROUPS:
        prefix = "hard_pair" if name == "hard" else f"{name}_pair"
        count = float(accumulator[name]["count"])
        correct = float(accumulator[name]["correct"])
        summary[f"{prefix}_count"] = count
        summary[f"{prefix}_acc"] = correct / max(count, 1.0)
    return summary


@torch.no_grad()
def evaluate(
    model: C3FaRNetSurfaceClassifier,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: dict[int, str],
    *,
    save_predictions_path: Path | None = None,
    save_logits_path: Path | None = None,
    logit_patch_rules: list[dict[str, Any]] | None = None,
    amp_dtype: torch.dtype | None = None,
    facet_teacher_logit_cache: Mapping[str, torch.Tensor] | None = None,
    facet_teacher_temperature: float = 2.0,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    logit_batches: list[np.ndarray] = []
    roughness_coral_logit_batches: list[np.ndarray] = []
    roughness_coral_presence: bool | None = None
    rfbt_diagnostic_names = (
        "transport_mass",
        "active_edge_fraction",
        "mean_edge_gate",
        "friction_reliability",
        "material_reliability",
        "roughness_reliability",
        "probability_mass_error",
    )
    rfbt_diagnostics: dict[str, list[torch.Tensor]] = {
        name: [] for name in rfbt_diagnostic_names
    }
    rfbt_presence: bool | None = None
    cfor_diagnostic_names = (
        "certified",
        "eligible",
        "exactly_one_factor",
        "rarer_runner_up",
        "joint_margin",
        "factor_margin",
        "probability_mass_error",
        "sorted_logit_error",
    )
    cfor_diagnostics: dict[str, list[torch.Tensor]] = {
        name: [] for name in cfor_diagnostic_names
    }
    cfor_presence: bool | None = None
    ngsc_diagnostic_names = (
        "local_evidence_rms",
        "positive_statistic_rms",
        "negative_statistic_rms",
        "delta_rms",
        "correction_rms",
        "correction_abs_max",
        "correction_class_sum_abs",
        "correction_nonzero_class_fraction",
    )
    ngsc_diagnostics: dict[str, list[torch.Tensor]] = {
        name: [] for name in ngsc_diagnostic_names
    }
    ngsc_presence: bool | None = None
    pcqt_diagnostic_names = (
        "energy_first_mean",
        "energy_second_mean",
        "support_fraction",
        "coherence_abs_mean",
        "coherence_abs_max",
        "coherence_unclamped_excess",
        "marginal_abs_mean",
        "minus_energy_min",
        "plus_energy_min",
        "coordinate_mean",
        "coordinate_abs_mean",
        "coordinate_abs_max",
        "writer_rms",
        "residual_scale",
        "residual_rms",
        "base_rms",
        "residual_to_base_rms_ratio",
    )
    pcqt_diagnostics: dict[str, list[torch.Tensor]] = {
        name: [] for name in pcqt_diagnostic_names
    }
    pcqt_presence: bool | None = None
    evaluated_paths: list[str] = []
    losses = []
    rows = []
    rho_values = []
    rho_label_indices: list[int] = []
    logit_patch_count = 0
    logit_patch_rule_hits: dict[str, int] = {}
    # Tournament diagnostics are sample/pair statistics, not batch statistics.
    # Accumulating the per-batch accuracy and dividing by the number of batches
    # makes the result depend on eval.batch_size (and weights a short final
    # batch as much as a full batch).  Preserve the sufficient statistics so
    # the reported values are identical for every evaluation partition.
    tournament_diagnostics = _new_tournament_diagnostic_accumulator()
    aort_diagnostic_samples = 0
    aort_stage_ratio_sum = torch.zeros(3, dtype=torch.float64)
    aort_stage_ratio_max = torch.zeros(3, dtype=torch.float64)
    aort_stage_update_rms_sum = torch.zeros(3, dtype=torch.float64)
    aort_calibration_residual_rms_sum = 0.0
    trace_bace_diagnostic_samples = 0
    trace_bace_correction_ratio_sum = torch.zeros(4, dtype=torch.float64)
    trace_bace_correction_ratio_max = torch.zeros(4, dtype=torch.float64)
    trace_bace_saturation_sum = torch.zeros(4, dtype=torch.float64)
    trace_bace_gate_sum = torch.zeros(4, dtype=torch.float64)
    trace_lmcr_diagnostics: dict[str, list[torch.Tensor]] = {
        "dispersion_rms": [],
        "correction_ratio": [],
        "moment_code_saturation": [],
        "context_code_saturation": [],
        "code_rms": [],
        "gate": [],
        "output_projection_frobenius": [],
    }
    tifr_diagnostic_names = (
        "theta_material_rms",
        "theta_material_spatial_std",
        "theta_material_saturation",
        "theta_roughness_rms",
        "theta_roughness_spatial_std",
        "theta_roughness_saturation",
    )
    tifr_diagnostics: dict[str, list[torch.Tensor]] = {
        name: [] for name in tifr_diagnostic_names
    }
    tifr_presence: bool | None = None
    forge_diagnostic_shapes = {
        "factor_output_rms": (3,),
        "pair_code_rms": (3,),
        "aggregate_relative_rms": (1,),
        "update_ratio": (1,),
        "saturation_fraction": (3,),
    }
    forge_diagnostics: dict[str, list[torch.Tensor]] = {
        key: [] for key in forge_diagnostic_shapes
    }
    forge_presence: bool | None = None
    scot_diagnostic_shapes = {
        "spatial_peak_mass": (3,),
        "spatial_entropy": (3,),
        "pairwise_support_js": (3,),
        "marginal_error": (3,),
        "transport_distance": (3,),
        "target_entropy": (1,),
        "consensus_relative_rms": (1,),
        "update_ratio": (1,),
    }
    scot_diagnostics: dict[str, list[torch.Tensor]] = {
        key: [] for key in scot_diagnostic_shapes
    }
    scot_presence: bool | None = None
    facet_diagnostic_shapes = {
        "facet_order_rms": (5, 4),
        "facet_ledger_rms": (5,),
        "facet_transport_ratio": (4,),
        "facet_transport_tau": (4,),
        "facet_transport_axis_norm": (4, 3),
        "facet_writeback_ratio": (),
        "facet_writeback_rms": (),
        "facet_output_projection_frobenius": (),
    }
    facet_diagnostics: dict[str, list[torch.Tensor]] = {
        key: [] for key in facet_diagnostic_shapes
    }
    facet_presence: bool | None = None
    facet_teacher_semantic_sums = {
        "trace_facet_teacher_valid_mass_mean": 0.0,
        "trace_facet_teacher_valid_cond_kl_mean": 0.0,
        "trace_facet_teacher_full120_kl_mean": 0.0,
        "trace_facet_teacher_top1_full120": 0.0,
        "trace_facet_teacher_top1_valid": 0.0,
    }
    facet_teacher_semantic_samples = 0
    facet_label_semantic_sums = {
        "trace_facet_valid_mass_mean": 0.0,
        "trace_facet_valid_cond_nll_mean": 0.0,
        "trace_facet_full120_nll_mean": 0.0,
        "trace_facet_label_top1_full120": 0.0,
        "trace_facet_label_top1_valid": 0.0,
    }
    facet_label_semantic_samples = 0
    facet_nll_identity_max_abs_error = 0.0
    if not math.isfinite(float(facet_teacher_temperature)) or float(
        facet_teacher_temperature
    ) <= 0.0:
        raise ValueError("facet_teacher_temperature must be finite and positive")
    for batch in tqdm(
        loader,
        desc="Validation",
        unit="batch",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    ):
        image = batch["image"].to(device, non_blocking=True)
        valid_mask = _batch_valid_mask(batch, device)
        observer_image = _batch_observer_image(batch, device)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            out = _forward_surface_model(
                model,
                image,
                return_aux=True,
                valid_mask=valid_mask,
                observer_image=observer_image,
            )
        raw_rfbt_diagnostics = out.get("rfbt_diagnostics")
        has_rfbt = isinstance(raw_rfbt_diagnostics, Mapping)
        if rfbt_presence is None:
            rfbt_presence = has_rfbt
        elif rfbt_presence != has_rfbt:
            raise RuntimeError("RFBT diagnostics appeared for only part of evaluation")
        if has_rfbt:
            if set(raw_rfbt_diagnostics) != set(rfbt_diagnostic_names):
                raise RuntimeError(
                    "RFBT diagnostic keys drifted: "
                    f"expected={sorted(rfbt_diagnostic_names)} "
                    f"got={sorted(raw_rfbt_diagnostics)}"
                )
            expected_shape = (int(label.numel()),)
            for name in rfbt_diagnostic_names:
                value = raw_rfbt_diagnostics[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        f"RFBT {name} must have shape {expected_shape}, got "
                        f"{getattr(value, 'shape', None)}"
                    )
                value = value.detach().float().cpu()
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"RFBT {name} contains NaN/Inf")
                rfbt_diagnostics[name].append(value)
        raw_cfor_diagnostics = out.get("cfor_diagnostics")
        has_cfor = isinstance(raw_cfor_diagnostics, Mapping)
        if cfor_presence is None:
            cfor_presence = has_cfor
        elif cfor_presence != has_cfor:
            raise RuntimeError("CFOR diagnostics appeared for only part of evaluation")
        if has_cfor:
            if set(raw_cfor_diagnostics) != set(cfor_diagnostic_names):
                raise RuntimeError(
                    "CFOR diagnostic keys drifted: "
                    f"expected={sorted(cfor_diagnostic_names)} "
                    f"got={sorted(raw_cfor_diagnostics)}"
                )
            expected_shape = (int(label.numel()),)
            for name in cfor_diagnostic_names:
                value = raw_cfor_diagnostics[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        f"CFOR {name} must have shape {expected_shape}, got "
                        f"{getattr(value, 'shape', None)}"
                    )
                value = value.detach().float().cpu()
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"CFOR {name} contains NaN/Inf")
                cfor_diagnostics[name].append(value)
        raw_ngsc_aux = out.get("nullspace_graph_readout_aux")
        has_ngsc = isinstance(raw_ngsc_aux, Mapping)
        if ngsc_presence is None:
            ngsc_presence = has_ngsc
        elif ngsc_presence != has_ngsc:
            raise RuntimeError(
                "NGSC diagnostics appeared for only part of evaluation"
            )
        if has_ngsc:
            if set(raw_ngsc_aux) != set(ngsc_diagnostic_names):
                raise RuntimeError(
                    "NGSC diagnostic keys drifted: "
                    f"expected={sorted(ngsc_diagnostic_names)} "
                    f"got={sorted(raw_ngsc_aux)}"
                )
            batch_size = int(label.numel())
            for name in ngsc_diagnostic_names:
                value = raw_ngsc_aux[name]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
                    batch_size,
                ):
                    raise RuntimeError(
                        f"NGSC {name} must have shape ({batch_size},), got "
                        f"{getattr(value, 'shape', None)}"
                    )
                value = value.detach().float().cpu()
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"NGSC {name} contains NaN/Inf")
                ngsc_diagnostics[name].append(value)
        backbone_aux = getattr(getattr(model, "backbone", None), "last_aux", None)
        if isinstance(backbone_aux, Mapping):
            pcqt_sources = {
                name: backbone_aux.get(f"pcqt_{name}")
                for name in pcqt_diagnostic_names
            }
            present_pcqt = {
                name: value
                for name, value in pcqt_sources.items()
                if value is not None
            }
            has_pcqt = bool(present_pcqt)
            if pcqt_presence is None:
                pcqt_presence = has_pcqt
            elif pcqt_presence != has_pcqt:
                raise RuntimeError(
                    "PCQT diagnostics appeared for only part of evaluation"
                )
            if present_pcqt and len(present_pcqt) != len(pcqt_sources):
                missing = sorted(set(pcqt_sources) - set(present_pcqt))
                raise RuntimeError(
                    "PCQT diagnostics are incomplete; missing last_aux keys: "
                    f"{missing}"
                )
            if present_pcqt:
                expected_shape = (int(label.numel()),)
                for name, value in present_pcqt.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != expected_shape
                    ):
                        raise RuntimeError(
                            "PCQT diagnostic must be a per-sample B tensor: "
                            f"pcqt_{name} expected={expected_shape} "
                            f"got={type(value).__name__} "
                            f"shape={getattr(value, 'shape', None)}"
                        )
                    value = value.detach().double().cpu()
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(
                            f"PCQT diagnostic contains NaN/Inf: pcqt_{name}"
                        )
                    pcqt_diagnostics[name].append(value)
        facet_joint_logits = out.get("facet_joint_logits")
        has_facet = isinstance(facet_joint_logits, torch.Tensor)
        if facet_presence is None:
            facet_presence = has_facet
        elif facet_presence != has_facet:
            raise RuntimeError(
                "facet_joint_logits appeared for only part of evaluation"
            )
        if has_facet:
            if not isinstance(backbone_aux, Mapping):
                raise RuntimeError(
                    "FACET evaluation requires backbone.last_aux diagnostics"
                )
            batch_size = int(label.numel())
            if tuple(facet_joint_logits.shape) != (batch_size, 120):
                raise RuntimeError(
                    "FACET evaluation joint logits must have shape Bx120; "
                    f"got {tuple(facet_joint_logits.shape)}"
                )
            for key, trailing_shape in facet_diagnostic_shapes.items():
                value = backbone_aux.get(key)
                expected_shape = (batch_size, *trailing_shape)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        "FACET diagnostic has an invalid shape: "
                        f"{key} expected={expected_shape} "
                        f"got={type(value).__name__} "
                        f"shape={getattr(value, 'shape', None)}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"FACET diagnostic contains NaN/Inf: {key}")
                facet_diagnostics[key].append(value.detach().double().cpu())

            class_to_factor = getattr(
                getattr(model, "spec", None),
                "class_to_factor",
                None,
            )
            if not isinstance(class_to_factor, torch.Tensor):
                raise RuntimeError(
                    "FACET validation requires model.spec.class_to_factor"
                )
            class_to_factor = class_to_factor.to(
                device=facet_joint_logits.device,
                dtype=torch.long,
            )
            if tuple(class_to_factor.shape) != (27, 3):
                raise RuntimeError(
                    "FACET validation class_to_factor must have shape 27x3; "
                    f"got {tuple(class_to_factor.shape)}"
                )
            valid_cells = (
                class_to_factor[:, 0] * 20
                + class_to_factor[:, 1] * 4
                + class_to_factor[:, 2]
            )
            if int(torch.unique(valid_cells).numel()) != 27:
                raise RuntimeError(
                    "FACET validation requires 27 unique legal factor cells"
                )

            # Ground-truth validation is deliberately independent of the S7
            # teacher.  It separates classification inside the 27 legal RSCD
            # cells from the probability mass assigned to the other 93 cells.
            joint_logits_f32 = facet_joint_logits.float()
            valid_logits_f32 = joint_logits_f32.index_select(1, valid_cells)
            true_cells = valid_cells.index_select(0, label.long())
            valid_cond_nll = F.cross_entropy(
                valid_logits_f32,
                label.long(),
                reduction="none",
            )
            full120_nll = F.cross_entropy(
                joint_logits_f32,
                true_cells,
                reduction="none",
            )
            log_valid_mass = (
                torch.logsumexp(valid_logits_f32, dim=1)
                - torch.logsumexp(joint_logits_f32, dim=1)
            )
            identity_error = (
                full120_nll - (valid_cond_nll - log_valid_mass)
            ).abs()
            if not bool(
                torch.isfinite(valid_cond_nll).all()
                and torch.isfinite(full120_nll).all()
                and torch.isfinite(log_valid_mass).all()
                and torch.isfinite(identity_error).all()
            ):
                raise RuntimeError(
                    "FACET ground-truth lattice diagnostics contain NaN/Inf"
                )
            facet_nll_identity_max_abs_error = max(
                facet_nll_identity_max_abs_error,
                float(identity_error.max().detach().cpu()),
            )
            facet_label_semantic_samples += batch_size
            facet_label_semantic_sums[
                "trace_facet_valid_mass_mean"
            ] += float(log_valid_mass.exp().sum().detach().cpu())
            facet_label_semantic_sums[
                "trace_facet_valid_cond_nll_mean"
            ] += float(valid_cond_nll.sum().detach().cpu())
            facet_label_semantic_sums[
                "trace_facet_full120_nll_mean"
            ] += float(full120_nll.sum().detach().cpu())
            facet_label_semantic_sums[
                "trace_facet_label_top1_full120"
            ] += float(
                joint_logits_f32.argmax(dim=1).eq(true_cells).sum().detach().cpu()
            )
            facet_label_semantic_sums[
                "trace_facet_label_top1_valid"
            ] += float(
                valid_logits_f32.argmax(dim=1).eq(label).sum().detach().cpu()
            )

            if facet_teacher_logit_cache is not None:
                teacher_logits, _ = cached_teacher_logits_for_batch(
                    facet_teacher_logit_cache,
                    batch["image_path"],
                    device=device,
                    dtype=torch.float32,
                    strict=True,
                    cache_name="facet_validation",
                    labels=label,
                )
                if teacher_logits is None:
                    raise RuntimeError(
                        "strict FACET validation teacher cache missed a batch"
                    )
                _, semantic_logs = _facet_joint_distillation_loss(
                    facet_joint_logits,
                    teacher_logits,
                    label,
                    class_to_factor,
                    {
                        "facet_joint_distill_weight": 1.0,
                        "facet_joint_distill_temperature": float(
                            facet_teacher_temperature
                        ),
                    },
                )
                facet_teacher_semantic_samples += batch_size
                teacher_semantic_key_map = {
                    "trace_facet_teacher_valid_mass_mean": (
                        "trace_facet_valid_mass_mean"
                    ),
                    "trace_facet_teacher_valid_cond_kl_mean": (
                        "trace_facet_valid_cond_kl_mean"
                    ),
                    "trace_facet_teacher_full120_kl_mean": (
                        "trace_facet_full120_kl_mean"
                    ),
                    "trace_facet_teacher_top1_full120": (
                        "trace_facet_teacher_top1_full120"
                    ),
                    "trace_facet_teacher_top1_valid": (
                        "trace_facet_teacher_top1_valid"
                    ),
                }
                for output_key, source_key in teacher_semantic_key_map.items():
                    facet_teacher_semantic_sums[output_key] += (
                        float(semantic_logs[source_key]) * batch_size
                    )
        if isinstance(backbone_aux, dict):
            forge_sources = {
                name: backbone_aux.get(name)
                for name in forge_diagnostic_shapes
            }
            # ``update_ratio`` is intentionally shared by several bounded
            # residual mechanisms.  Treat it as a member only after a
            # mechanism-specific signature key is present, otherwise an SCoT
            # run would be misdiagnosed as an incomplete FORGE run (and vice
            # versa).
            forge_signature = any(
                forge_sources[name] is not None
                for name in forge_diagnostic_shapes
                if name != "update_ratio"
            )
            present_forge = {
                name: value
                for name, value in forge_sources.items()
                if value is not None and forge_signature
            }
            has_forge = bool(present_forge)
            if forge_presence is None:
                forge_presence = has_forge
            elif forge_presence != has_forge:
                raise RuntimeError(
                    "FORGE diagnostics appeared for only part of evaluation"
                )
            if present_forge and len(present_forge) != len(forge_sources):
                missing = sorted(set(forge_sources) - set(present_forge))
                raise RuntimeError(
                    "FORGE diagnostics are incomplete; missing last_aux keys: "
                    f"{missing}"
                )
            if present_forge:
                batch_size = int(label.numel())
                for name, trailing_shape in forge_diagnostic_shapes.items():
                    value = present_forge[name]
                    expected_shape = (batch_size, *trailing_shape)
                    if (
                        not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != expected_shape
                    ):
                        raise RuntimeError(
                            "FORGE diagnostic has an invalid shape: "
                            f"{name} expected={expected_shape} "
                            f"got={type(value).__name__} "
                            f"shape={getattr(value, 'shape', None)}"
                        )
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(
                            f"FORGE diagnostic contains NaN/Inf: {name}"
                        )
                    if bool(value.lt(0.0).any()):
                        raise RuntimeError(
                            f"FORGE diagnostic must be non-negative: {name}"
                        )
                    if (
                        name == "saturation_fraction"
                        and bool(value.gt(1.0).any())
                    ):
                        raise RuntimeError(
                            "FORGE saturation_fraction exceeds one"
                        )
                    forge_diagnostics[name].append(
                        value.detach().double().cpu()
                    )
            spatial_mass = backbone_aux.get("spatial_mass")
            spatial_peak_mass = None
            if isinstance(spatial_mass, torch.Tensor):
                if (
                    spatial_mass.ndim != 3
                    or int(spatial_mass.shape[0]) != int(label.numel())
                    or int(spatial_mass.shape[1]) != 3
                ):
                    raise RuntimeError(
                        "SCoT spatial_mass must have shape Bx3xP; "
                        f"got {tuple(spatial_mass.shape)}"
                    )
                if not bool(torch.isfinite(spatial_mass).all()):
                    raise RuntimeError("SCoT spatial_mass contains NaN/Inf")
                if bool(spatial_mass.lt(0.0).any()):
                    raise RuntimeError("SCoT spatial_mass must be non-negative")
                spatial_peak_mass = spatial_mass.max(dim=-1).values
            scot_sources = {
                name: (
                    spatial_peak_mass
                    if name == "spatial_peak_mass"
                    else backbone_aux.get(name)
                )
                for name in scot_diagnostic_shapes
            }
            scot_signature = any(
                scot_sources[name] is not None
                for name in scot_diagnostic_shapes
                if name != "update_ratio"
            )
            present_scot = {
                name: value
                for name, value in scot_sources.items()
                if value is not None and scot_signature
            }
            has_scot = bool(present_scot)
            if scot_presence is None:
                scot_presence = has_scot
            elif scot_presence != has_scot:
                raise RuntimeError(
                    "SCoT diagnostics appeared for only part of evaluation"
                )
            if present_scot and len(present_scot) != len(scot_sources):
                missing = sorted(set(scot_sources) - set(present_scot))
                raise RuntimeError(
                    "SCoT diagnostics are incomplete; missing last_aux keys: "
                    f"{missing}"
                )
            if present_scot:
                batch_size = int(label.numel())
                for name, trailing_shape in scot_diagnostic_shapes.items():
                    value = present_scot[name]
                    expected_shape = (batch_size, *trailing_shape)
                    if (
                        not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != expected_shape
                    ):
                        raise RuntimeError(
                            "SCoT diagnostic has an invalid shape: "
                            f"{name} expected={expected_shape} "
                            f"got={type(value).__name__} "
                            f"shape={getattr(value, 'shape', None)}"
                        )
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(
                            f"SCoT diagnostic contains NaN/Inf: {name}"
                        )
                    if bool(value.lt(0.0).any()):
                        raise RuntimeError(
                            f"SCoT diagnostic must be non-negative: {name}"
                        )
                    if (
                        name in {"spatial_entropy", "target_entropy"}
                        and bool(value.gt(1.000001).any())
                    ):
                        raise RuntimeError(
                            f"SCoT normalized entropy exceeds one: {name}"
                        )
                    scot_diagnostics[name].append(
                        value.detach().double().cpu()
                    )
            tifr_sources = {
                name: backbone_aux.get(name) for name in tifr_diagnostic_names
            }
            present_tifr = {
                name: value
                for name, value in tifr_sources.items()
                if value is not None
            }
            has_tifr = bool(present_tifr)
            if tifr_presence is None:
                tifr_presence = has_tifr
            elif tifr_presence != has_tifr:
                raise RuntimeError(
                    "TIFR diagnostics appeared for only part of evaluation"
                )
            if present_tifr and len(present_tifr) != len(tifr_sources):
                missing = sorted(set(tifr_sources) - set(present_tifr))
                raise RuntimeError(
                    "TIFR diagnostics are incomplete; missing last_aux keys: "
                    f"{missing}"
                )
            if present_tifr:
                expected_shape = (int(label.numel()),)
                for name, value in present_tifr.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != expected_shape
                    ):
                        raise RuntimeError(
                            "TIFR diagnostic must be a per-sample B tensor: "
                            f"{name} expected={expected_shape} "
                            f"got={type(value).__name__} "
                            f"shape={getattr(value, 'shape', None)}"
                        )
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(
                            f"TIFR diagnostic contains NaN/Inf: {name}"
                        )
                    if bool(value.lt(0.0).any()):
                        raise RuntimeError(
                            f"TIFR diagnostic must be non-negative: {name}"
                        )
                    if name.endswith("_saturation") and bool(value.gt(1.0).any()):
                        raise RuntimeError(
                            f"TIFR saturation diagnostic exceeds one: {name}"
                        )
                    tifr_diagnostics[name].append(
                        value.detach().double().cpu()
                    )
            lmcr_sources = {
                name: backbone_aux.get(f"lmcr_{name}")
                for name in trace_lmcr_diagnostics
            }
            present_lmcr = {
                name: value
                for name, value in lmcr_sources.items()
                if value is not None
            }
            if present_lmcr and len(present_lmcr) != len(lmcr_sources):
                missing = sorted(set(lmcr_sources) - set(present_lmcr))
                raise RuntimeError(
                    "TRACE-LMCR diagnostics are incomplete; missing "
                    f"last_aux keys: {missing}"
                )
            if present_lmcr:
                expected_shape = (int(label.numel()), 4)
                for name, value in present_lmcr.items():
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                        raise RuntimeError(
                            "TRACE-LMCR diagnostic must be a Bx4 tensor: "
                            f"lmcr_{name} expected={expected_shape} "
                            f"got={type(value).__name__} "
                            f"shape={getattr(value, 'shape', None)}"
                        )
                    trace_lmcr_diagnostics[name].append(
                        value.detach().double().cpu()
                    )
            correction_ratio = backbone_aux.get("agreement_correction_ratio")
            saturation_fraction = backbone_aux.get(
                "agreement_saturation_fraction"
            )
            agreement_gate = backbone_aux.get("agreement_gate")
            if (
                isinstance(correction_ratio, torch.Tensor)
                and isinstance(saturation_fraction, torch.Tensor)
                and isinstance(agreement_gate, torch.Tensor)
                and correction_ratio.ndim == 2
                and correction_ratio.shape == saturation_fraction.shape
                and correction_ratio.shape == agreement_gate.shape
                and int(correction_ratio.shape[1]) == 4
            ):
                ratio_cpu = correction_ratio.detach().double().cpu()
                saturation_cpu = saturation_fraction.detach().double().cpu()
                gate_cpu = agreement_gate.detach().double().cpu()
                trace_bace_diagnostic_samples += int(ratio_cpu.shape[0])
                trace_bace_correction_ratio_sum += ratio_cpu.sum(dim=0)
                trace_bace_correction_ratio_max = torch.maximum(
                    trace_bace_correction_ratio_max,
                    ratio_cpu.max(dim=0).values,
                )
                trace_bace_saturation_sum += saturation_cpu.sum(dim=0)
                trace_bace_gate_sum += gate_cpu.sum(dim=0)
        aort_diagnostics = out.get("aort_spatial_diagnostics")
        if isinstance(aort_diagnostics, dict):
            update_rms = aort_diagnostics.get("stage_update_rms")
            carrier_rms = aort_diagnostics.get("stage_downsample_rms")
            calibration_rms = aort_diagnostics.get(
                "calibration_residual_rms"
            )
            if (
                isinstance(update_rms, torch.Tensor)
                and isinstance(carrier_rms, torch.Tensor)
                and update_rms.ndim == 2
                and carrier_rms.shape == update_rms.shape
                and int(update_rms.shape[1]) == 3
            ):
                update_cpu = update_rms.detach().double().cpu()
                carrier_cpu = carrier_rms.detach().double().cpu()
                ratio_cpu = torch.where(
                    carrier_cpu > 0.0,
                    update_cpu / carrier_cpu.clamp_min(1.0e-30),
                    torch.zeros_like(update_cpu),
                )
                aort_diagnostic_samples += int(update_cpu.shape[0])
                aort_stage_ratio_sum += ratio_cpu.sum(dim=0)
                aort_stage_ratio_max = torch.maximum(
                    aort_stage_ratio_max,
                    ratio_cpu.max(dim=0).values,
                )
                aort_stage_update_rms_sum += update_cpu.sum(dim=0)
                if (
                    isinstance(calibration_rms, torch.Tensor)
                    and calibration_rms.numel() == update_cpu.shape[0]
                ):
                    aort_calibration_residual_rms_sum += float(
                        calibration_rms.detach().double().sum().cpu()
                    )
        # Accumulate losses, probabilities and exported logits in fp32 so AMP
        # changes throughput rather than the metric definition.
        logits = out["logits"].float()
        raw_roughness_logits = out.get("roughness_coral_logits")
        has_roughness_logits = isinstance(raw_roughness_logits, torch.Tensor)
        if roughness_coral_presence is None:
            roughness_coral_presence = has_roughness_logits
        elif roughness_coral_presence != has_roughness_logits:
            raise RuntimeError(
                "roughness_coral_logits appeared for only part of evaluation"
            )
        if has_roughness_logits:
            if tuple(raw_roughness_logits.shape) != (int(label.numel()), 2):
                raise RuntimeError(
                    "roughness_coral_logits must have shape [B, 2], got "
                    f"{tuple(raw_roughness_logits.shape)}"
                )
            roughness_coral_logit_batches.append(
                raw_roughness_logits.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
        if logit_patch_rules:
            logits, patch_logs = apply_pareto_safe_logit_patch(
                logits,
                logit_patch_rules,
                idx_to_class,
            )
            logit_patch_count += int(patch_logs.get("count", 0))
            for key, value in patch_logs.get("rule_hits", {}).items():
                logit_patch_rule_hits[key] = logit_patch_rule_hits.get(key, 0) + int(value)
        loss = F.cross_entropy(logits, label)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        logit_batches.append(logits.detach().cpu().numpy().astype(np.float32, copy=True))
        y_true.extend(label.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
        evaluated_paths.extend(str(path) for path in batch["image_path"])
        losses.append(float(loss.detach().cpu()) * int(label.numel()))
        if isinstance(out.get("rho_roughness"), torch.Tensor):
            rho_values.extend(out["rho_roughness"].detach().float().cpu().view(-1).numpy().tolist())
            rho_label_indices.extend(label.detach().cpu().numpy().astype(int).tolist())
        boundary_logits = out.get("boundary_logits", {})
        if isinstance(boundary_logits, dict):
            boundary_logits = {
                key: value.float() if isinstance(value, torch.Tensor) else value
                for key, value in boundary_logits.items()
            }
        _, tour_logs = mechanism_routed_tournament_loss(
            logits,
            label,
            boundary_logits,
            model.spec,
        )
        _accumulate_tournament_diagnostics(tournament_diagnostics, tour_logs)
        if save_predictions_path is not None:
            for path, true_idx, pred_idx, confidence in zip(batch["image_path"], y_true[-len(label):], y_pred[-len(label):], conf.detach().cpu().tolist(), strict=True):
                rows.append(
                    {
                        "image_path": str(path),
                        "true_label": idx_to_class[int(true_idx)],
                        "pred_label": idx_to_class[int(pred_idx)],
                        "confidence": float(confidence),
                    }
                )
    labels = list(range(len(idx_to_class)))
    expected_samples = int(len(loader.dataset))
    if len(evaluated_paths) != expected_samples:
        raise RuntimeError(
            "Evaluation sample count changed from the frozen dataset: "
            f"expected={expected_samples} evaluated={len(evaluated_paths)}"
        )
    canonical_evaluated_paths = [
        path.strip().replace("\\", "/").casefold() for path in evaluated_paths
    ]
    unique_evaluated_paths = len(set(canonical_evaluated_paths))
    if unique_evaluated_paths != len(canonical_evaluated_paths):
        raise RuntimeError(
            "Evaluation contains duplicate image paths; refusing to report a "
            f"frozen-protocol result: evaluated={len(canonical_evaluated_paths)} "
            f"unique={unique_evaluated_paths}"
        )
    target_names = [idx_to_class[i] for i in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    total_seen = len(y_true)
    factor_summary = factor_confusion_summary(y_true, y_pred, model.spec, idx_to_class)
    hard_class_names = [
        "water_concrete_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_severe",
        "water_asphalt_slight",
        "dry_concrete_slight",
    ]
    hard_scores = [float(report[name]["f1-score"]) for name in hard_class_names if name in report]
    wcs_report = report.get("water_concrete_slight", {})
    rho_slice = rho_group_summary(rho_values, rho_label_indices, model.spec, idx_to_class)
    per_class_f1 = [
        (name, float(report.get(name, {}).get("f1-score", 0.0)))
        for name in target_names
    ]
    per_class_f1_sorted = sorted(per_class_f1, key=lambda item: (item[1], item[0]))
    bottom_five = per_class_f1_sorted[: min(5, len(per_class_f1_sorted))]
    min_class_name, min_class_f1 = (
        per_class_f1_sorted[0] if per_class_f1_sorted else ("", 0.0)
    )
    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    logits_array = np.concatenate(logit_batches, axis=0)
    if logits_array.shape != (total_seen, len(labels)):
        raise RuntimeError(
            "evaluation logit collection changed shape: "
            f"expected={(total_seen, len(labels))} got={tuple(logits_array.shape)}"
        )
    # Calibration is part of the frozen validation evidence for classifier-side
    # logit readouts.  Compute it directly from the already collected FP32
    # logits, without fitting a temperature or consulting another split.
    # Float64 here prevents an extreme-but-finite FP32 logit from underflowing
    # before normalization.  The multiclass Brier definition is the mean sum
    # of squared probability errors; ECE uses 15 fixed equal-width top-label
    # confidence bins.
    calibration_logits = logits_array.astype(np.float64, copy=False)
    calibration_logits = calibration_logits - calibration_logits.max(
        axis=1,
        keepdims=True,
    )
    calibration_exp = np.exp(calibration_logits)
    calibration_probabilities = calibration_exp / calibration_exp.sum(
        axis=1,
        keepdims=True,
    )
    calibration_rows = np.arange(total_seen, dtype=np.int64)
    calibration_true_probability = calibration_probabilities[
        calibration_rows,
        y_true_array,
    ]
    calibration_nll = float(
        -np.log(np.clip(calibration_true_probability, 1.0e-300, 1.0)).mean()
    )
    calibration_target = np.zeros_like(calibration_probabilities)
    calibration_target[calibration_rows, y_true_array] = 1.0
    calibration_brier = float(
        np.square(calibration_probabilities - calibration_target)
        .sum(axis=1)
        .mean()
    )
    calibration_confidence = calibration_probabilities.max(axis=1)
    calibration_prediction = calibration_probabilities.argmax(axis=1)
    calibration_correct = calibration_prediction == y_true_array
    calibration_bin_count = 15
    calibration_bin = np.minimum(
        (calibration_confidence * calibration_bin_count).astype(np.int64),
        calibration_bin_count - 1,
    )
    calibration_ece = 0.0
    for calibration_bin_index in range(calibration_bin_count):
        calibration_mask = calibration_bin == calibration_bin_index
        calibration_count = int(calibration_mask.sum())
        if calibration_count == 0:
            continue
        calibration_ece += (
            float(calibration_count) / float(max(total_seen, 1))
        ) * abs(
            float(calibration_correct[calibration_mask].mean())
            - float(calibration_confidence[calibration_mask].mean())
        )
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    evaluation_hashes = _ordered_evaluation_hashes(
        evaluated_paths,
        y_true_array,
        y_pred_array,
        logits_array,
    )
    roughness_coral_array: np.ndarray | None = None
    roughness_coral_eval_logs: dict[str, float] = {}
    if roughness_coral_logit_batches:
        roughness_coral_array = np.concatenate(
            roughness_coral_logit_batches,
            axis=0,
        ).astype(np.float32, copy=False)
        if roughness_coral_array.shape != (total_seen, 2):
            raise RuntimeError(
                "evaluation roughness CORAL collection changed shape: "
                f"expected={(total_seen, 2)} "
                f"got={tuple(roughness_coral_array.shape)}"
            )
        _, roughness_coral_eval_logs = coral_roughness_loss(
            torch.from_numpy(roughness_coral_array),
            torch.from_numpy(y_true_array),
            model.spec,
        )
        roughness_bytes = np.ascontiguousarray(
            roughness_coral_array,
            dtype=np.dtype("<f4"),
        ).tobytes(order="C")
        evaluation_hashes["roughness_coral_logits_fp32_sha256"] = (
            hashlib.sha256(roughness_bytes).hexdigest()
        )

    def pair_accuracy(left: str, right: str) -> float:
        left_idx = class_to_idx.get(left)
        right_idx = class_to_idx.get(right)
        if left_idx is None or right_idx is None:
            return 0.0
        mask = (y_true_array == int(left_idx)) | (y_true_array == int(right_idx))
        if not bool(mask.any()):
            return 0.0
        return float((y_true_array[mask] == y_pred_array[mask]).mean())

    wcs_rough = _pair_classification_diagnostics(
        y_true_array,
        y_pred_array,
        logits_array,
        left_idx=class_to_idx.get("water_concrete_slight", -1),
        right_idx=class_to_idx.get("water_concrete_severe", -1),
    )
    wcs_wet = _pair_classification_diagnostics(
        y_true_array,
        y_pred_array,
        logits_array,
        left_idx=class_to_idx.get("water_concrete_slight", -1),
        right_idx=class_to_idx.get("wet_concrete_slight", -1),
    )

    summary = {
        "loss": float(sum(losses) / max(total_seen, 1)),
        "nll": calibration_nll,
        "brier_score": calibration_brier,
        "ece_15": float(calibration_ece),
        "top1": float(accuracy_score(y_true, y_pred)),
        "mean_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mean_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "num_samples": int(total_seen),
        "unique_image_paths": int(unique_evaluated_paths),
        "num_classes": int(len(labels)),
        "hard_class_mean_f1": float(np.mean(hard_scores)) if hard_scores else 0.0,
        "min_class_f1": float(min_class_f1),
        "min_class_name": str(min_class_name),
        "bottom5_mean_f1": float(np.mean([score for _, score in bottom_five]))
        if bottom_five
        else 0.0,
        "bottom5_classes": [name for name, _ in bottom_five],
        "per_class_f1": {name: score for name, score in per_class_f1},
        "water_concrete_slight_precision": float(wcs_report.get("precision", 0.0)),
        "water_concrete_slight_recall": float(wcs_report.get("recall", 0.0)),
        "water_concrete_slight_f1": float(report.get("water_concrete_slight", {}).get("f1-score", 0.0)),
        "water_concrete_slight_vs_water_concrete_severe_pair_accuracy": pair_accuracy(
            "water_concrete_slight",
            "water_concrete_severe",
        ),
        "water_concrete_slight_vs_wet_concrete_slight_pair_accuracy": pair_accuracy(
            "water_concrete_slight",
            "wet_concrete_slight",
        ),
        "water_concrete_slight_vs_water_concrete_severe_binary_logit_accuracy": float(
            wcs_rough["binary_logit_accuracy"]
        ),
        "water_concrete_slight_vs_water_concrete_severe_direct_confusions": int(
            wcs_rough["direct_27way_confusions"]
        ),
        "water_concrete_slight_to_water_concrete_severe_count": int(
            wcs_rough["left_to_right_27way"]
        ),
        "water_concrete_severe_to_water_concrete_slight_count": int(
            wcs_rough["right_to_left_27way"]
        ),
        "water_concrete_slight_vs_water_concrete_severe_third_class_errors": int(
            wcs_rough["third_class_errors"]
        ),
        "water_concrete_slight_vs_wet_concrete_slight_binary_logit_accuracy": float(
            wcs_wet["binary_logit_accuracy"]
        ),
        "water_concrete_slight_vs_wet_concrete_slight_direct_confusions": int(
            wcs_wet["direct_27way_confusions"]
        ),
        "water_concrete_slight_to_wet_concrete_slight_count": int(
            wcs_wet["left_to_right_27way"]
        ),
        "wet_concrete_slight_to_water_concrete_slight_count": int(
            wcs_wet["right_to_left_27way"]
        ),
        "water_concrete_slight_vs_wet_concrete_slight_third_class_errors": int(
            wcs_wet["third_class_errors"]
        ),
        "rho_R_mean": float(np.mean(rho_values)) if rho_values else 0.0,
        "head_type": str(getattr(model, "head_type", "")),
        "dryvor_enabled": bool(getattr(model, "dry_concrete_roughness_vor_residual", None) is not None),
        "logits_after_dryvor": bool(getattr(model, "dry_concrete_roughness_vor_residual", None) is not None),
        "boundary_use_physics_feature": bool(getattr(model, "boundary_use_physics_feature", False)),
        "pareto_safe_logit_patch_enabled": bool(logit_patch_rules),
        "pareto_safe_logit_patch_count": int(logit_patch_count),
        **evaluation_hashes,
    }
    if roughness_coral_array is not None:
        summary.update(roughness_coral_eval_logs)
        scale = getattr(model, "roughness_aux_scale", None)
        if isinstance(scale, torch.Tensor) and scale.numel() == 1:
            summary["roughness_aux_scale"] = float(scale.detach().cpu())
        roughness_head = getattr(model, "roughness_coral_head", None)
        if roughness_head is None:
            roughness_head = getattr(model, "arcq_roughness_coral_head", None)
        thresholds = getattr(roughness_head, "thresholds", None)
        if callable(thresholds):
            summary["roughness_coral_thresholds"] = (
                thresholds().detach().float().cpu().tolist()
            )
    for key, value in sorted(logit_patch_rule_hits.items()):
        summary[f"pareto_safe_logit_patch_hits/{key}"] = int(value)
    summary.update(rho_slice)
    summary.update(_finalize_tournament_diagnostics(tournament_diagnostics))
    summary.update(factor_summary["summary"])
    if rfbt_presence:
        rfbt_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in rfbt_diagnostics.items()
        }
        for name, values in rfbt_stacked.items():
            summary[f"rfbt_{name}_mean"] = float(values.mean())
        summary["rfbt_transport_mass_p95"] = float(
            torch.quantile(rfbt_stacked["transport_mass"], 0.95)
        )
        summary["rfbt_probability_mass_error_max"] = float(
            rfbt_stacked["probability_mass_error"].max()
        )
    if cfor_presence:
        cfor_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in cfor_diagnostics.items()
        }
        if any(
            tuple(values.shape) != (total_seen,)
            for values in cfor_stacked.values()
        ):
            raise RuntimeError("CFOR diagnostic sample counts diverged")
        summary["cfor_diagnostic_samples"] = int(total_seen)
        summary["cfor_certified_count"] = int(cfor_stacked["certified"].sum())
        summary["cfor_certified_fraction"] = float(cfor_stacked["certified"].mean())
        summary["cfor_eligible_fraction"] = float(cfor_stacked["eligible"].mean())
        summary["cfor_exactly_one_factor_fraction"] = float(
            cfor_stacked["exactly_one_factor"].mean()
        )
        summary["cfor_rarer_runner_up_fraction"] = float(
            cfor_stacked["rarer_runner_up"].mean()
        )
        certified_mask = cfor_stacked["certified"].bool()
        if bool(certified_mask.any()):
            summary["cfor_joint_margin_certified_mean"] = float(
                cfor_stacked["joint_margin"][certified_mask].mean()
            )
            summary["cfor_factor_margin_certified_mean"] = float(
                cfor_stacked["factor_margin"][certified_mask].mean()
            )
        else:
            summary["cfor_joint_margin_certified_mean"] = 0.0
            summary["cfor_factor_margin_certified_mean"] = 0.0
        summary["cfor_probability_mass_error_max"] = float(
            cfor_stacked["probability_mass_error"].max()
        )
        summary["cfor_sorted_logit_error_max"] = float(
            cfor_stacked["sorted_logit_error"].max()
        )
    if pcqt_presence:
        pcqt_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in pcqt_diagnostics.items()
        }
        if any(
            tuple(values.shape) != (total_seen,)
            for values in pcqt_stacked.values()
        ):
            raise RuntimeError("PCQT diagnostic sample counts diverged")
        summary["pcqt_diagnostic_samples"] = int(total_seen)
        for name, values in pcqt_stacked.items():
            prefix = f"pcqt_{name}"
            summary[f"{prefix}_mean"] = float(values.mean())
            summary[f"{prefix}_p50"] = float(torch.quantile(values, 0.50))
            summary[f"{prefix}_p95"] = float(torch.quantile(values, 0.95))
            summary[f"{prefix}_min"] = float(values.min())
            summary[f"{prefix}_max"] = float(values.max())

        # These class-conditional support summaries are validation diagnostics,
        # not task-specific routing.  They make the fail-closed PCQT gate check
        # whether the quotient is numerically supported on the exact hard
        # classes that motivated the transition, without changing logits.
        for class_name in (
            "water_concrete_slight",
            "water_concrete_severe",
            "wet_concrete_slight",
            "wet_concrete_severe",
        ):
            class_index = class_to_idx.get(class_name)
            if class_index is None:
                raise RuntimeError(
                    f"PCQT support audit is missing class {class_name!r}"
                )
            class_mask = torch.from_numpy(
                y_true_array == int(class_index)
            )
            if not bool(class_mask.any()):
                raise RuntimeError(
                    f"PCQT support audit has no validation samples for {class_name!r}"
                )
            class_support = pcqt_stacked["support_fraction"][class_mask]
            summary[
                f"pcqt_support_fraction_mean_{class_name}"
            ] = float(class_support.mean())
    if ngsc_presence:
        ngsc_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in ngsc_diagnostics.items()
        }
        if any(
            tuple(values.shape) != (total_seen,)
            for values in ngsc_stacked.values()
        ):
            raise RuntimeError("NGSC diagnostic sample counts diverged")
        summary["ngsc_diagnostic_samples"] = int(total_seen)
        for name, values in ngsc_stacked.items():
            summary[f"ngsc_{name}_mean"] = float(values.mean())
            summary[f"ngsc_{name}_p50"] = float(
                torch.quantile(values, 0.50)
            )
            summary[f"ngsc_{name}_p95"] = float(
                torch.quantile(values, 0.95)
            )
            summary[f"ngsc_{name}_max"] = float(values.max())
    if forge_presence:
        forge_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in forge_diagnostics.items()
        }
        if any(
            tuple(values.shape)
            != (total_seen, *forge_diagnostic_shapes[name])
            for name, values in forge_stacked.items()
        ):
            raise RuntimeError("FORGE diagnostic sample counts diverged")
        summary["forge_diagnostic_samples"] = int(total_seen)
        for name, values in forge_stacked.items():
            prefix = f"forge_{name}"
            mean = values.mean(dim=0)
            p50 = torch.quantile(values, 0.50, dim=0)
            p95 = torch.quantile(values, 0.95, dim=0)
            maximum = values.max(dim=0).values
            if int(values.shape[1]) == 1:
                summary[f"{prefix}_mean"] = float(mean.item())
                summary[f"{prefix}_p50"] = float(p50.item())
                summary[f"{prefix}_p95"] = float(p95.item())
                summary[f"{prefix}_max"] = float(maximum.item())
            else:
                summary[f"{prefix}_mean"] = mean.tolist()
                summary[f"{prefix}_p50"] = p50.tolist()
                summary[f"{prefix}_p95"] = p95.tolist()
                summary[f"{prefix}_max"] = maximum.tolist()
    if scot_presence:
        scot_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in scot_diagnostics.items()
        }
        if any(
            tuple(values.shape)
            != (total_seen, *scot_diagnostic_shapes[name])
            for name, values in scot_stacked.items()
        ):
            raise RuntimeError("SCoT diagnostic sample counts diverged")
        summary["scot_diagnostic_samples"] = int(total_seen)
        for name, values in scot_stacked.items():
            prefix = f"scot_{name}"
            mean = values.mean(dim=0)
            p50 = torch.quantile(values, 0.50, dim=0)
            p95 = torch.quantile(values, 0.95, dim=0)
            maximum = values.max(dim=0).values
            if int(values.shape[1]) == 1:
                summary[f"{prefix}_mean"] = float(mean.item())
                summary[f"{prefix}_p50"] = float(p50.item())
                summary[f"{prefix}_p95"] = float(p95.item())
                summary[f"{prefix}_max"] = float(maximum.item())
            else:
                summary[f"{prefix}_mean"] = mean.tolist()
                summary[f"{prefix}_p50"] = p50.tolist()
                summary[f"{prefix}_p95"] = p95.tolist()
                summary[f"{prefix}_max"] = maximum.tolist()
    if tifr_presence:
        tifr_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in tifr_diagnostics.items()
        }
        if any(
            tuple(values.shape) != (total_seen,)
            for values in tifr_stacked.values()
        ):
            raise RuntimeError("TIFR diagnostic sample counts diverged")
        summary["tifr_diagnostic_samples"] = int(total_seen)
        for name, values in tifr_stacked.items():
            summary[f"tifr_{name}_mean"] = float(values.mean())
            summary[f"tifr_{name}_p50"] = float(torch.quantile(values, 0.50))
            summary[f"tifr_{name}_p95"] = float(torch.quantile(values, 0.95))
            summary[f"tifr_{name}_max"] = float(values.max())
    if aort_diagnostic_samples > 0:
        denominator = float(aort_diagnostic_samples)
        summary.update(
            {
                "aort_diagnostic_samples": int(aort_diagnostic_samples),
                "aort_stage_update_ratio_mean": (
                    aort_stage_ratio_sum / denominator
                ).tolist(),
                "aort_stage_update_ratio_max": aort_stage_ratio_max.tolist(),
                "aort_stage_update_rms_mean": (
                    aort_stage_update_rms_sum / denominator
                ).tolist(),
                "aort_calibration_residual_rms_mean": float(
                    aort_calibration_residual_rms_sum / denominator
                ),
            }
        )
    if trace_bace_diagnostic_samples > 0:
        denominator = float(trace_bace_diagnostic_samples)
        summary.update(
            {
                "trace_bace_diagnostic_samples": int(
                    trace_bace_diagnostic_samples
                ),
                "trace_bace_correction_ratio_mean": (
                    trace_bace_correction_ratio_sum / denominator
                ).tolist(),
                "trace_bace_correction_ratio_max": (
                    trace_bace_correction_ratio_max.tolist()
                ),
                "trace_bace_saturation_fraction_mean": (
                    trace_bace_saturation_sum / denominator
                ).tolist(),
                "trace_bace_gate_mean": (
                    trace_bace_gate_sum / denominator
                ).tolist(),
            }
        )
    if trace_lmcr_diagnostics["correction_ratio"]:
        lmcr_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in trace_lmcr_diagnostics.items()
        }
        lmcr_samples = int(lmcr_stacked["correction_ratio"].shape[0])
        if any(
            tuple(values.shape) != (lmcr_samples, 4)
            for values in lmcr_stacked.values()
        ):
            raise RuntimeError("TRACE-LMCR diagnostic sample counts diverged")
        summary["trace_lmcr_diagnostic_samples"] = lmcr_samples
        for name, values in lmcr_stacked.items():
            summary[f"trace_lmcr_{name}_mean"] = values.mean(dim=0).tolist()
            summary[f"trace_lmcr_{name}_p50"] = torch.quantile(
                values,
                0.50,
                dim=0,
            ).tolist()
            summary[f"trace_lmcr_{name}_p95"] = torch.quantile(
                values,
                0.95,
                dim=0,
            ).tolist()
            summary[f"trace_lmcr_{name}_max"] = values.max(dim=0).values.tolist()
    if facet_presence:
        facet_stacked = {
            name: torch.cat(values, dim=0)
            for name, values in facet_diagnostics.items()
        }
        if any(
            int(values.shape[0]) != total_seen
            for values in facet_stacked.values()
        ):
            raise RuntimeError("FACET diagnostic sample counts diverged")
        order = facet_stacked["facet_order_rms"]
        ledger = facet_stacked["facet_ledger_rms"]
        transport = facet_stacked["facet_transport_ratio"]
        tau = facet_stacked["facet_transport_tau"]
        axis_norm = facet_stacked["facet_transport_axis_norm"]
        writeback_ratio = facet_stacked["facet_writeback_ratio"]
        writeback_rms = facet_stacked["facet_writeback_rms"]
        output_projection = facet_stacked[
            "facet_output_projection_frobenius"
        ]
        summary.update(
            {
                "trace_facet_diagnostic_samples": int(total_seen),
                "trace_facet_order_rms_mean": order.mean(dim=0).tolist(),
                "trace_facet_ledger_rms_mean": ledger.mean(dim=0).tolist(),
                "trace_facet_transport_ratio_mean": transport.mean(dim=0).tolist(),
                "trace_facet_transport_ratio_p95": torch.quantile(
                    transport, 0.95, dim=0
                ).tolist(),
                "trace_facet_transport_ratio_max": transport.max(dim=0).values.tolist(),
                "trace_facet_transport_tau_mean": tau.mean(dim=0).tolist(),
                "trace_facet_transport_tau_max": tau.max(dim=0).values.tolist(),
                "trace_facet_transport_axis_norm_mean": axis_norm.mean(dim=0).tolist(),
                "trace_facet_transport_axis_norm_max": axis_norm.max(dim=0).values.tolist(),
                "trace_facet_writeback_ratio_mean": float(writeback_ratio.mean()),
                "trace_facet_writeback_ratio_p95": float(
                    torch.quantile(writeback_ratio, 0.95)
                ),
                "trace_facet_writeback_ratio_max": float(writeback_ratio.max()),
                "trace_facet_writeback_rms_mean": float(writeback_rms.mean()),
                "trace_facet_writeback_rms_p95": float(
                    torch.quantile(writeback_rms, 0.95)
                ),
                "trace_facet_writeback_rms_max": float(writeback_rms.max()),
                "trace_facet_output_projection_frobenius_mean": float(
                    output_projection.mean()
                ),
            }
        )
        if facet_label_semantic_samples != total_seen:
            raise RuntimeError(
                "FACET ground-truth semantic sample count diverged: "
                f"labels={facet_label_semantic_samples} evaluated={total_seen}"
            )
        summary.update(
            {
                key: value / max(facet_label_semantic_samples, 1)
                for key, value in facet_label_semantic_sums.items()
            }
        )
        summary["trace_facet_nll_identity_max_abs_error"] = float(
            facet_nll_identity_max_abs_error
        )
        if facet_teacher_logit_cache is not None:
            if facet_teacher_semantic_samples != total_seen:
                raise RuntimeError(
                    "FACET validation teacher sample count diverged: "
                    f"teacher={facet_teacher_semantic_samples} "
                    f"evaluated={total_seen}"
                )
            summary.update(
                {
                    key: value / max(facet_teacher_semantic_samples, 1)
                    for key, value in facet_teacher_semantic_sums.items()
                }
            )
    elif facet_teacher_logit_cache is not None:
        raise RuntimeError(
            "a FACET validation teacher cache was supplied to a model without "
            "facet_joint_logits"
        )
    if save_predictions_path is not None:
        save_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(save_predictions_path, index=False, encoding="utf-8")
    if save_logits_path is not None:
        save_logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            save_logits_path,
            np.ascontiguousarray(logits_array, dtype=np.dtype("<f4")),
            allow_pickle=False,
        )
        if roughness_coral_array is not None:
            roughness_path = save_logits_path.with_name(
                f"roughness_coral_{save_logits_path.name}"
            )
            np.save(
                roughness_path,
                np.ascontiguousarray(
                    roughness_coral_array,
                    dtype=np.dtype("<f4"),
                ),
                allow_pickle=False,
            )
    return {
        "summary": summary,
        "classification_report": report,
        "factor_confusion_summary": factor_summary,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def load_pareto_safe_logit_patch_rules(path: Path | None) -> list[dict[str, Any]]:
    """Load validation-accepted RSCD hard-edge logit patch rules."""

    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"logit patch rules file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_rules = payload.get("accepted_rules", [])
    elif isinstance(payload, list):
        raw_rules = payload
    else:
        raise ValueError(f"unsupported logit patch rule payload in {path}")
    rules: list[dict[str, Any]] = []
    for item in raw_rules:
        if not isinstance(item, dict):
            continue
        rule = item.get("rule_raw", item)
        if isinstance(rule, dict) and {"source", "target", "topk", "margin", "delta"}.issubset(rule):
            rules.append(
                {
                    "source": str(rule["source"]),
                    "target": str(rule["target"]),
                    "topk": int(rule["topk"]),
                    "margin": float(rule["margin"]),
                    "delta": float(rule["delta"]),
                }
            )
    return rules


def apply_pareto_safe_logit_patch(
    logits: torch.Tensor,
    rules: list[dict[str, Any]],
    idx_to_class: dict[int, str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply split-validation-accepted RSCD hard-edge logit corrections."""

    if not rules:
        return logits, {"count": 0, "rule_hits": {}}
    class_to_idx = {canonical_class_label(name): int(idx) for idx, name in idx_to_class.items()}
    out = logits
    total_hits = 0
    rule_hits: dict[str, int] = {}
    for rule in rules:
        source_name = canonical_class_label(str(rule["source"]))
        target_name = canonical_class_label(str(rule["target"]))
        if source_name not in class_to_idx or target_name not in class_to_idx:
            continue
        source = int(class_to_idx[source_name])
        target = int(class_to_idx[target_name])
        topk = max(1, min(int(rule.get("topk", 2)), int(out.shape[1])))
        margin = float(rule.get("margin", 0.0))
        delta = float(rule.get("delta", 0.0))
        pred = out.argmax(dim=1)
        order = torch.argsort(out, dim=1, descending=True)
        in_topk = order[:, :topk].eq(int(target)).any(dim=1)
        close = (out[:, source] - out[:, target]) <= margin
        mask = pred.eq(int(source)) & in_topk & close
        hit_count = int(mask.detach().sum().cpu())
        if hit_count <= 0:
            continue
        if out is logits:
            out = logits.clone()
        out[mask, target] = out[mask, target] + delta
        out[mask, source] = out[mask, source] - delta * 0.25
        key = f"{source_name}->{target_name}"
        rule_hits[key] = rule_hits.get(key, 0) + hit_count
        total_hits += hit_count
    return out, {"count": int(total_hits), "rule_hits": rule_hits}


def rho_group_summary(
    rho_values: list[float],
    labels: list[int],
    spec: RSCDFactorSpec,
    idx_to_class: dict[int, str],
) -> dict[str, float]:
    """Summarize roughness visibility reliability by friction state and hard class."""

    if not rho_values or not labels:
        empty = {"rho_R_mean_water": 0.0, "rho_R_mean_wet": 0.0, "rho_R_mean_dry": 0.0}
        for name in ("water_concrete_slight", "water_concrete_severe", "wet_concrete_slight", "dry_concrete_slight"):
            empty[f"rho_R_mean_{name}"] = 0.0
            empty[f"rho_R_gap_dry_minus_{name}"] = 0.0
        return empty
    rho = np.asarray(rho_values, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    factors = spec.class_to_factor.numpy()
    friction = factors[label_arr, 0]
    friction_names = list(FACTOR_LABELS["friction"])

    def mean_for(name: str) -> float:
        if name not in friction_names:
            return 0.0
        idx = friction_names.index(name)
        mask = friction == idx
        return float(rho[mask].mean()) if bool(mask.any()) else 0.0

    return {
        "rho_R_mean_water": mean_for("water"),
        "rho_R_mean_wet": mean_for("wet"),
        "rho_R_mean_dry": mean_for("dry"),
        **rho_hard_class_summary(rho, label_arr, idx_to_class, dry_reference=mean_for("dry")),
    }


def rho_hard_class_summary(
    rho: np.ndarray,
    labels: np.ndarray,
    idx_to_class: dict[int, str],
    *,
    dry_reference: float,
) -> dict[str, float]:
    """Return rho_R means for the RSCD hard classes named in the goal."""

    class_to_idx = {str(name): int(idx) for idx, name in idx_to_class.items()}
    target_names = (
        "water_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_slight",
        "dry_concrete_slight",
    )
    out: dict[str, float] = {}
    for name in target_names:
        idx = class_to_idx.get(name)
        if idx is None:
            mean_value = 0.0
        else:
            mask = labels == int(idx)
            mean_value = float(rho[mask].mean()) if bool(mask.any()) else 0.0
        out[f"rho_R_mean_{name}"] = mean_value
        out[f"rho_R_gap_dry_minus_{name}"] = float(dry_reference - mean_value)
    return out


def factor_confusion_summary(
    y_true: list[int],
    y_pred: list[int],
    spec: RSCDFactorSpec,
    idx_to_class: dict[int, str],
) -> dict[str, Any]:
    factors = spec.class_to_factor.numpy()
    errors = 0
    friction_error = 0
    material_error = 0
    roughness_error = 0
    axis_valid = {axis: 0 for axis in FACTOR_AXES}
    axis_correct = {axis: 0 for axis in FACTOR_AXES}
    rows = []
    for true_idx, pred_idx in zip(y_true, y_pred, strict=True):
        t = factors[int(true_idx)]
        p = factors[int(pred_idx)]
        axis_diff = {}
        for axis_i, axis in enumerate(FACTOR_AXES):
            valid = int(t[axis_i]) >= 0 and int(p[axis_i]) >= 0
            if valid:
                axis_valid[axis] += 1
                if int(t[axis_i]) == int(p[axis_i]):
                    axis_correct[axis] += 1
            axis_diff[axis] = bool(valid and int(t[axis_i]) != int(p[axis_i]))
        if int(true_idx) != int(pred_idx):
            errors += 1
            friction_error += int(axis_diff["friction"])
            material_error += int(axis_diff["material"])
            roughness_error += int(axis_diff["roughness"])
            rows.append(
                {
                    "true": idx_to_class[int(true_idx)],
                    "pred": idx_to_class[int(pred_idx)],
                    **{f"{axis}_error": axis_diff[axis] for axis in FACTOR_AXES},
                }
            )
    summary = {
        "friction_acc": axis_correct["friction"] / max(axis_valid["friction"], 1),
        "material_acc": axis_correct["material"] / max(axis_valid["material"], 1),
        "roughness_acc": axis_correct["roughness"] / max(axis_valid["roughness"], 1),
        "friction_error_share": friction_error / max(errors, 1),
        "material_error_share": material_error / max(errors, 1),
        "roughness_error_share": roughness_error / max(errors, 1),
        "num_errors": int(errors),
    }
    return {"summary": summary, "error_rows": rows}


def write_outputs(out_dir: Path, metrics: dict[str, Any], idx_to_class: dict[int, str], split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k not in {"y_true", "y_pred"}}
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    report = metrics["classification_report"]
    with (out_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for idx in range(len(idx_to_class)):
            name = idx_to_class[idx]
            item = report.get(name, {})
            w.writerow([name, item.get("precision", 0.0), item.get("recall", 0.0), item.get("f1-score", 0.0), item.get("support", 0)])
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(idx_to_class))))
    pd.DataFrame(cm, index=[idx_to_class[i] for i in range(len(idx_to_class))], columns=[idx_to_class[i] for i in range(len(idx_to_class))]).to_csv(
        out_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    (out_dir / "factor_confusion_summary.json").write_text(
        json.dumps(metrics["factor_confusion_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_hard_pair_metrics(out_dir / "hard_pair_metrics.csv", metrics, idx_to_class)
    write_wcs_diagnosis(out_dir / "water_concrete_slight_diagnosis.json", metrics, idx_to_class)


def materialize_eval_checkpoint_aliases(out_dir: Path, checkpoint: Path, *, split: str) -> None:
    """Expose the evaluated checkpoint under the standard train/eval artifact names.

    Training naturally writes best/last checkpoints. Evaluation consumes an
    existing checkpoint, so the honest equivalent is an alias to the source
    checkpoint plus a small provenance file. On the same volume this uses a
    hard link and costs no extra checkpoint storage.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(checkpoint).resolve()
    provenance = {
        "split": str(split),
        "source_checkpoint": str(source),
        "artifact_role": "evaluation alias of the checkpoint passed to validate.py/test.py",
    }
    (out_dir / "checkpoint_used.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("best_checkpoint.pth", "last_checkpoint.pth"):
        target = out_dir / name
        if target.exists():
            continue
        try:
            os.link(source, target)
            continue
        except OSError as link_error:
            try:
                shutil.copy2(source, target)
                continue
            except OSError as copy_error:
                torch.save(
                    {
                        **provenance,
                        "warning": "source checkpoint could not be hard-linked or copied",
                        "link_error": repr(link_error),
                        "copy_error": repr(copy_error),
                    },
                    target,
                )


def write_hard_pair_metrics(path: Path, metrics: dict[str, Any], idx_to_class: dict[int, str]) -> None:
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    spec = build_rscd_factor_spec(class_to_idx)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["left", "right", "axis", "boundary", "pair_samples", "pair_acc"])
        for pair in spec.hard_pairs:
            mask = (y_true == pair.left) | (y_true == pair.right)
            if not mask.any():
                continue
            ok = y_true[mask] == y_pred[mask]
            w.writerow([idx_to_class[pair.left], idx_to_class[pair.right], pair.axis, pair.boundary, int(mask.sum()), float(ok.mean())])


def write_wcs_diagnosis(path: Path, metrics: dict[str, Any], idx_to_class: dict[int, str]) -> None:
    class_to_idx = {name: idx for idx, name in idx_to_class.items()}
    spec = build_rscd_factor_spec(class_to_idx)
    target = class_to_idx.get("water_concrete_slight")
    payload: dict[str, Any] = {"class": "water_concrete_slight", "present": target is not None}
    if target is not None:
        y_true = np.asarray(metrics["y_true"], dtype=int)
        y_pred = np.asarray(metrics["y_pred"], dtype=int)
        mask = y_true == int(target)
        report = metrics["classification_report"].get("water_concrete_slight", {})
        # Exclude the correct class before truncating.  Otherwise a frequent
        # correct prediction consumes one of the eight slots even though it is
        # discarded below, and ``top_confused_classes`` can contain only seven
        # actual confusions.
        counts = pd.Series(y_pred[mask & (y_pred != int(target))]).value_counts().head(8)
        factor_error_counts = _target_factor_error_counts(y_true, y_pred, int(target), spec)
        support = int(mask.sum())
        misclassified = int((y_true[mask] != y_pred[mask]).sum())
        payload.update(
            {
                "precision": float(report.get("precision", 0.0)),
                "recall": float(report.get("recall", 0.0)),
                "f1": float(report.get("f1-score", 0.0)),
                "support": support,
                "misclassified": misclassified,
                "top_confused_classes": [
                    {"pred": idx_to_class[int(idx)], "count": int(count)}
                    for idx, count in counts.items()
                    if int(idx) != int(target)
                ],
                "roughness_error_count": int(factor_error_counts["roughness"]),
                "friction_error_count": int(factor_error_counts["friction"]),
                "material_error_count": int(factor_error_counts["material"]),
                # A single wrong prediction can disagree on multiple factors,
                # so these per-factor shares are intentionally non-exclusive.
                "roughness_error_rate_over_support": float(
                    factor_error_counts["roughness"] / max(support, 1)
                ),
                "friction_error_rate_over_support": float(
                    factor_error_counts["friction"] / max(support, 1)
                ),
                "material_error_rate_over_support": float(
                    factor_error_counts["material"] / max(support, 1)
                ),
                "roughness_error_share_among_misclassified": float(
                    factor_error_counts["roughness"] / max(misclassified, 1)
                ),
                "friction_error_share_among_misclassified": float(
                    factor_error_counts["friction"] / max(misclassified, 1)
                ),
                "material_error_share_among_misclassified": float(
                    factor_error_counts["material"] / max(misclassified, 1)
                ),
            }
        )
        factor_summary = metrics["factor_confusion_summary"]["summary"]
        payload.update(
            {
                "global_roughness_error_share": factor_summary.get(
                    "roughness_error_share",
                    0.0,
                ),
                "global_friction_error_share": factor_summary.get(
                    "friction_error_share",
                    0.0,
                ),
                "global_material_error_share": factor_summary.get(
                    "material_error_share",
                    0.0,
                ),
                "rho_R_mean": metrics["summary"].get("rho_R_mean", 0.0),
                "rho_R_mean_water": metrics["summary"].get("rho_R_mean_water", 0.0),
                "rho_R_mean_wet": metrics["summary"].get("rho_R_mean_wet", 0.0),
                "rho_R_mean_dry": metrics["summary"].get("rho_R_mean_dry", 0.0),
                "rho_R_mean_water_concrete_slight": metrics["summary"].get("rho_R_mean_water_concrete_slight", 0.0),
                "rho_R_mean_water_concrete_severe": metrics["summary"].get("rho_R_mean_water_concrete_severe", 0.0),
                "rho_R_mean_wet_concrete_slight": metrics["summary"].get("rho_R_mean_wet_concrete_slight", 0.0),
                "rho_R_mean_dry_concrete_slight": metrics["summary"].get("rho_R_mean_dry_concrete_slight", 0.0),
                "rho_R_gap_dry_minus_water_concrete_slight": metrics["summary"].get(
                    "rho_R_gap_dry_minus_water_concrete_slight",
                    0.0,
                ),
                "rho_R_gap_dry_minus_water_concrete_severe": metrics["summary"].get(
                    "rho_R_gap_dry_minus_water_concrete_severe",
                    0.0,
                ),
                "rho_R_gap_dry_minus_wet_concrete_slight": metrics["summary"].get(
                    "rho_R_gap_dry_minus_wet_concrete_slight",
                    0.0,
                ),
                "tournament_pair_accuracy_involving_water_concrete_slight": _wcs_pair_acc(metrics, class_to_idx, idx_to_class),
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _target_factor_error_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_idx: int,
    spec: RSCDFactorSpec,
) -> dict[str, int]:
    factors = spec.class_to_factor.numpy()
    mask = (y_true == int(target_idx)) & (y_pred != int(target_idx))
    counts = {axis: 0 for axis in FACTOR_AXES}
    if not bool(mask.any()):
        return counts
    target_factors = factors[y_true[mask]]
    pred_factors = factors[y_pred[mask]]
    for axis_i, axis in enumerate(FACTOR_AXES):
        valid = (target_factors[:, axis_i] >= 0) & (pred_factors[:, axis_i] >= 0)
        counts[axis] = int(((target_factors[:, axis_i] != pred_factors[:, axis_i]) & valid).sum())
    return counts


def _wcs_pair_acc(metrics: dict[str, Any], class_to_idx: dict[str, int], idx_to_class: dict[int, str]) -> float:
    target = class_to_idx.get("water_concrete_slight")
    if target is None:
        return 0.0
    spec = build_rscd_factor_spec(class_to_idx)
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    accs = []
    for pair in spec.hard_pairs:
        if pair.left != target and pair.right != target:
            continue
        mask = (y_true == pair.left) | (y_true == pair.right)
        if mask.any():
            accs.append(float((y_true[mask] == y_pred[mask]).mean()))
    return float(np.mean(accs)) if accs else 0.0


def _validate_grit_step_resume_config(
    saved_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> None:
    """Reject shape-compatible checkpoints from a different GRIT control.

    GRIT controls deliberately use identical parameter names and shapes.  A
    strict ``state_dict`` load therefore cannot distinguish the full
    gauge-relational transition from its no-sign and raster controls.  Clean
    development configs disable automatic resume; this guard protects an
    explicitly enabled continuation.
    """

    current_model = current_config.get("model", {})
    if not isinstance(current_model, dict):
        return
    backbone_name = str(current_model.get("backbone", "")).strip().lower()
    if not backbone_name.startswith("grit_road_s"):
        return
    if not isinstance(saved_config, dict):
        raise RuntimeError(
            "GRIT step checkpoint has no saved config; mechanism-safe resume "
            "cannot be verified"
        )
    saved_model = saved_config.get("model")
    if not isinstance(saved_model, dict) or saved_model != current_model:
        raise RuntimeError(
            "GRIT step checkpoint model config does not match the current "
            "run; refusing a shape-compatible cross-mechanism resume"
        )
    saved_train = saved_config.get("train", {})
    current_train = current_config.get("train", {})
    saved_loss = saved_config.get("loss", {})
    current_loss = current_config.get("loss", {})
    if not isinstance(saved_train, dict) or not isinstance(current_train, dict):
        raise RuntimeError("GRIT step checkpoint has no comparable train config")
    if not isinstance(saved_loss, dict) or not isinstance(current_loss, dict):
        raise RuntimeError("GRIT step checkpoint has no comparable loss config")

    def runtime_contract(
        train: dict[str, Any],
        loss: dict[str, Any],
    ) -> dict[str, Any]:
        allow_tf32 = bool(train.get("allow_tf32", False))
        return {
            "amp": bool(train.get("amp", True)),
            "amp_dtype": str(train.get("amp_dtype", "auto")).strip().lower(),
            "eval_amp": bool(train.get("eval_amp", False)),
            "fused_adamw": bool(train.get("fused_adamw", False)),
            "cudnn_benchmark": bool(train.get("cudnn_benchmark", False)),
            "cudnn_deterministic": bool(
                train.get("cudnn_deterministic", False)
            ),
            "allow_tf32": allow_tf32,
            "float32_matmul_precision": str(
                train.get(
                    "float32_matmul_precision",
                    "high" if allow_tf32 else "highest",
                )
            ).strip().lower(),
            "pcgrad_enabled": bool(
                loss.get(
                    "rscd_pcgrad_enabled",
                    train.get("rscd_pcgrad_enabled", False),
                )
            ),
        }

    saved_runtime = runtime_contract(saved_train, saved_loss)
    current_runtime = runtime_contract(current_train, current_loss)
    if saved_runtime != current_runtime:
        raise RuntimeError(
            "GRIT step checkpoint runtime/precision contract does not match "
            "the current run; exact resume requires identical AMP, PCGrad, "
            "fused-optimizer and CUDA math settings: "
            f"checkpoint={saved_runtime} current={current_runtime}"
        )


def _validate_step_resume_config(
    saved_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> None:
    """Apply opt-in semantic contracts before loading a step checkpoint.

    A strict state-dict load cannot detect fixed, non-persistent forward
    hyperparameters.  In particular, the ARCQ factor-aux and factor-product
    arms have deliberately identical trainable parameters while their product
    scales live in the model config.  Opt-in experiments can therefore bind a
    resume to the exact model, data and loss sections that created it.
    """

    _validate_grit_step_resume_config(saved_config, current_config)
    train = current_config.get("train", {})
    if not isinstance(train, dict):
        return
    requirements = {
        "model": bool(train.get("require_step_checkpoint_model_config_match", False)),
        "data": bool(train.get("require_step_checkpoint_data_config_match", False)),
        "loss": bool(train.get("require_step_checkpoint_loss_config_match", False)),
    }
    if not any(requirements.values()):
        return
    if not isinstance(saved_config, dict):
        raise RuntimeError(
            "step checkpoint has no saved config; the requested semantic "
            "resume contract cannot be verified"
        )
    for section, required in requirements.items():
        if not required:
            continue
        saved_section = saved_config.get(section)
        current_section = current_config.get(section)
        if not isinstance(saved_section, dict) or not isinstance(
            current_section,
            dict,
        ):
            raise RuntimeError(
                f"step checkpoint has no comparable {section} config"
            )
        if saved_section != current_section:
            raise RuntimeError(
                f"step checkpoint {section} config does not match the current "
                "run; refusing a shape-compatible cross-arm resume"
            )


def _resolve_stop_after_epoch(train_cfg: dict[str, Any]) -> tuple[int, int]:
    """Return the full LR horizon and an optional earlier frozen gate epoch."""

    configured_epochs = int(train_cfg.get("epochs", 1))
    if configured_epochs < 1:
        raise ValueError("train.epochs must be positive")
    raw_stop_after_epoch = train_cfg.get("stop_after_epoch", configured_epochs)
    if isinstance(raw_stop_after_epoch, bool):
        raise TypeError("train.stop_after_epoch cannot be a boolean")
    stop_after_epoch = int(raw_stop_after_epoch)
    if stop_after_epoch != raw_stop_after_epoch or not (
        1 <= stop_after_epoch <= configured_epochs
    ):
        raise ValueError(
            "train.stop_after_epoch must be an integer in "
            f"[1,{configured_epochs}]; got {raw_stop_after_epoch!r}"
        )
    return configured_epochs, stop_after_epoch


def run_train(
    config_path: Path,
    *,
    seed_override: int | None = None,
    output_dir_override: Path | None = None,
) -> None:
    """训练总入口：从配置/断点建立完整状态并运行到目标 epoch。

    精确续训的顺序不能打乱：

    1. 加载并锁定 config；
    2. 优先发现 ``last_step_checkpoint.pth``；
    3. 核对 seed、model/data/loss、epoch 内 step 与梯度累积边界；
    4. 建立同长度 DataLoader，并核对样本数与 batch size；
    5. strict=True 加载模型，恢复 optimizer/scaler/RNG；
    6. 跳过本 epoch 已完成的 batch，从下一步继续；
    7. 定期原子保存步级断点，每个完整 epoch 后验证并更新 best。

    用户通常不直接调用本函数，而是运行 ``scripts/run_training.py``，因为后者
    还负责跨电脑路径映射和首次断点放置。
    """
    # -------- 第 1 阶段：解析配置并确定输出位置 --------
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if output_dir_override is not None:
        cfg["output_dir"] = str(output_dir_override)
    set_seed(int(cfg.get("seed", 79)))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg["train"]
    run_test_after_train = bool(train_cfg.get("run_test_after_train", True))
    configured_epochs, stop_after_epoch = _resolve_stop_after_epoch(train_cfg)
    # -------- 第 2 阶段：优先读取 epoch 内步级断点 --------
    # 与 last_checkpoint（完整 epoch）相比，它保留了 Epoch 50 step 2500 的进度。
    step_resume_state: dict[str, Any] | None = None
    step_resume_path = Path(str(train_cfg.get("resume_step_checkpoint_from") or (out_dir / "last_step_checkpoint.pth")))
    if bool(train_cfg.get("resume_step_checkpoint", False)) and step_resume_path.exists():
        step_resume_state = torch.load(step_resume_path, map_location="cpu", weights_only=False)
        resume_step = int(step_resume_state.get("step", 0) or 0)
        resume_total_steps = int(step_resume_state.get("total_steps", 0) or 0)
        saved_config = step_resume_state.get("config")
        # model/data/loss 任一处不一致都立即停止，避免生成不可比较的混合实验。
        _validate_step_resume_config(saved_config, cfg)
        if isinstance(saved_config, dict):
            saved_seed = int(saved_config.get("seed", cfg.get("seed", 79)))
            current_seed = int(cfg.get("seed", 79))
            if saved_seed != current_seed:
                raise RuntimeError(
                    "step checkpoint seed does not match the current run: "
                    f"checkpoint={saved_seed} current={current_seed}"
                )
        resume_epoch_complete = bool(
            resume_total_steps > 0 and resume_step >= resume_total_steps
        )
        grad_accum_steps = max(
            int(train_cfg.get("grad_accum_steps", 1)),
            1,
        )
        # checkpoint 必须落在梯度累积窗口边界；否则未保存的梯度会造成静默偏差。
        if not resume_epoch_complete and resume_step % grad_accum_steps != 0:
            raise RuntimeError(
                "cannot safely resume a legacy step checkpoint saved inside a "
                "gradient-accumulation window because pending gradients were "
                f"not serialized: step={resume_step} accum={grad_accum_steps}"
            )
        cfg["train"]["_resume_start_step"] = resume_step
        cfg["train"]["_resume_epoch_training_complete"] = resume_epoch_complete
        cfg["train"]["_resume_step_checkpoint_path"] = str(step_resume_path)
        cfg["train"]["_resume_train_partial"] = dict(
            step_resume_state.get("train_partial", {}) or {}
        )
        print(
            "Resuming in-epoch training checkpoint: "
            f"{step_resume_path} step={resume_step}/{resume_total_steps} "
            f"epoch_training_complete={cfg['train']['_resume_epoch_training_complete']}"
        )
    else:
        cfg["train"]["_resume_start_step"] = 0
        cfg["train"]["_resume_epoch_training_complete"] = False
        cfg["train"]["_resume_train_partial"] = {}
    start_epoch = int(step_resume_state.get("epoch", 1) if step_resume_state is not None else 1)
    cfg["train"]["_current_epoch"] = int(start_epoch)
    # -------- 第 3 阶段：建立固定 RSCD-27 标签映射与 DataLoader --------
    data = cfg["data"]
    manifests = [
        _runtime_path(data["train_manifest"]),
        _runtime_path(data["val_manifest"]),
    ]
    if run_test_after_train:
        manifests.append(_runtime_path(data["test_manifest"]))
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    if step_resume_state is not None:
        # 换电脑允许路径变化，但不允许 DataLoader 的数学意义发生变化。
        saved_class_to_idx = step_resume_state.get("class_to_idx")
        if saved_class_to_idx and dict(saved_class_to_idx) != class_to_idx:
            raise RuntimeError(
                "step checkpoint class_to_idx does not match the fixed RSCD-27 "
                "mapping resolved for the current run"
            )
    (out_dir / "label_factor_sanity.txt").write_text(sanity_summary(class_to_idx), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    train_loader, val_loader, test_loader = build_loaders(
        cfg,
        class_to_idx,
        include_test=run_test_after_train,
    )
    if step_resume_state is not None:
        saved_total_steps = int(step_resume_state.get("total_steps", 0) or 0)
        resumed_total_steps = int(cfg["train"]["_resume_start_step"]) + len(
            train_loader
        )
        if saved_total_steps > 0 and resumed_total_steps != saved_total_steps:
            raise RuntimeError(
                "step checkpoint no longer matches the current loader length: "
                f"checkpoint_total_steps={saved_total_steps} "
                f"resumed_total_steps={resumed_total_steps}"
            )
        saved_dataset_size = step_resume_state.get("train_dataset_size")
        if (
            saved_dataset_size is not None
            and int(saved_dataset_size) != len(train_loader.dataset)
        ):
            raise RuntimeError(
                "step checkpoint training dataset size does not match the "
                "current manifest/config: "
                f"checkpoint={int(saved_dataset_size)} "
                f"current={len(train_loader.dataset)}"
            )
        saved_batch_size = step_resume_state.get("train_batch_size")
        current_batch_size = int(
            getattr(train_loader, "batch_size", None)
            or getattr(
                getattr(train_loader, "batch_sampler", None),
                "batch_size",
                0,
            )
            or 0
        )
        if (
            saved_batch_size is not None
            and int(saved_batch_size) != current_batch_size
        ):
            raise RuntimeError(
                "step checkpoint batch size does not match the current config: "
                f"checkpoint={int(saved_batch_size)} current={current_batch_size}"
            )
    # -------- 第 4 阶段：配置 CUDA/BF16 并严格恢复模型 --------
    device = resolve_device()
    runtime_audit = _configure_cuda_runtime(train_cfg, device)
    (out_dir / "runtime_audit.json").write_text(
        json.dumps(runtime_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    model = build_model(cfg, class_to_idx).to(device)
    if step_resume_state is not None:
        # strict=True：少一个或多一个参数键都报错，不进行模糊迁移。
        model.load_state_dict(step_resume_state["model"], strict=True)
        _reject_legacy_arcq_step_resume(model)
        load_audit = dict(
            step_resume_state.get("load_audit")
            or step_resume_state.get("config", {})
            .get("train", {})
            .get("_load_audit", {})
            or {
                "mode": "strict_step_resume",
                "path": str(step_resume_path),
            }
        )
    else:
        if bool(train_cfg.get("resume_strict_model_state", False)):
            resume_from = train_cfg.get("resume_from")
            if not resume_from:
                raise RuntimeError(
                    "train.resume_strict_model_state=true requires train.resume_from"
                )
            incompatible_transfer_options = {
                name: train_cfg.get(name)
                for name in (
                    "s7_parent_exact_resume",
                    "resume_skip_prefixes",
                    "resume_immutable_prefixes",
                    "resume_required_loaded_keys",
                    "resume_allowed_missing_prefixes",
                    "resume_allowed_skipped_prefixes",
                )
                if train_cfg.get(name) not in (None, False, [], ())
            }
            if incompatible_transfer_options:
                raise RuntimeError(
                    "strict model-state resume cannot be combined with migration "
                    "or flexible-transfer options: "
                    + ", ".join(sorted(incompatible_transfer_options))
                )
            load_audit = strict_model_state_load(
                model,
                str(resume_from),
                expected_sha256=str(train_cfg.get("resume_expected_sha256", "")),
                expected_class_to_idx=class_to_idx,
                expected_state_key_count=train_cfg.get(
                    "resume_expected_loaded_key_count"
                ),
                expected_classifier_type=str(
                    cfg.get("model", {}).get("classifier_type", "")
                ),
            )
        elif bool(train_cfg.get("s7_parent_exact_resume", False)):
            resume_from = train_cfg.get("resume_from")
            if not resume_from:
                raise RuntimeError(
                    "train.s7_parent_exact_resume=true requires train.resume_from"
                )
            incompatible_transfer_options = {
                name: train_cfg.get(name)
                for name in (
                    "resume_skip_prefixes",
                    "resume_immutable_prefixes",
                    "resume_required_loaded_keys",
                    "resume_allowed_missing_prefixes",
                    "resume_allowed_skipped_prefixes",
                    "resume_expected_loaded_key_count",
                )
                if train_cfg.get(name) not in (None, [], ())
            }
            if incompatible_transfer_options:
                raise RuntimeError(
                    "S7 parent-exact resume cannot be combined with flexible "
                    "transfer options: "
                    + ", ".join(sorted(incompatible_transfer_options))
                )
            strict_audit = load_s7_parent_exact_state(
                model,
                str(resume_from),
                expected_class_to_idx=class_to_idx,
                expected_sha256=str(
                    train_cfg.get(
                        "resume_expected_sha256",
                        S7_PARENT_CHECKPOINT_SHA256,
                    )
                ),
                allowed_new_prefixes=train_cfg.get(
                    "s7_parent_exact_allowed_new_prefixes"
                ),
                expected_source_key_count=int(
                    train_cfg.get("s7_parent_exact_expected_source_key_count", 679)
                ),
            )
            load_audit = asdict(strict_audit)
            load_audit["path"] = strict_audit.checkpoint_path
            loaded_count = int(strict_audit.loaded_parent_key_count)
            load_audit["mode"] = (
                f"s7_family_exact_{loaded_count}_of_{loaded_count}"
            )
        else:
            load_audit = flexible_load(
                model,
                cfg["train"].get("resume_from"),
                skip_prefixes=cfg["train"].get("resume_skip_prefixes"),
                expected_sha256=cfg["train"].get("resume_expected_sha256"),
                immutable_prefixes=cfg["train"].get("resume_immutable_prefixes"),
                expected_class_to_idx=class_to_idx,
                require_class_to_idx=bool(
                    cfg["train"].get("resume_require_class_to_idx", False)
                ),
                required_loaded_keys=cfg["train"].get("resume_required_loaded_keys"),
                allowed_missing_prefixes=cfg["train"].get(
                    "resume_allowed_missing_prefixes"
                ),
                allowed_skipped_prefixes=cfg["train"].get(
                    "resume_allowed_skipped_prefixes"
                ),
                expected_loaded_key_count=cfg["train"].get(
                    "resume_expected_loaded_key_count"
                ),
            )
            load_audit["mode"] = (
                "flexible_weights_transfer"
                if load_audit.get("path")
                else "fresh_initialization"
            )
    head_initialization_audit = getattr(model, "head_initialization_audit", None)
    if isinstance(head_initialization_audit, dict):
        load_audit["head_initialization"] = copy.deepcopy(
            head_initialization_audit
        )
    parent_checkpoint_audit = getattr(model, "parent_checkpoint_audit", None)
    if not isinstance(parent_checkpoint_audit, dict):
        parent_checkpoint_audit = getattr(
            getattr(model, "backbone", None),
            "parent_checkpoint_audit",
            None,
        )
    if isinstance(parent_checkpoint_audit, dict) and parent_checkpoint_audit:
        # Constructor-owned parent loading is separate from resume_from.  Keep
        # both provenances in one audit so "fresh_initialization" can never be
        # misread as "no released parent was loaded".
        load_audit["constructor_parent_checkpoint"] = copy.deepcopy(
            parent_checkpoint_audit
        )
    cfg["train"]["_load_audit"] = load_audit
    (out_dir / "load_audit.json").write_text(
        json.dumps(load_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # The first resolved-config write happens before model construction so data
    # failures are still auditable. Rewrite it now to include checkpoint-load
    # provenance discovered while constructing the model.
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    apply_trainable_prefixes(model, cfg["train"].get("trainable_prefixes"))
    ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
    if ema is not None and step_resume_state is not None and "ema" in step_resume_state:
        ema.load_state_dict(step_resume_state["ema"])
    data_cfg = cfg.get("data", {})
    cached_teacher_fast_path = bool(
        train_cfg.get("cached_teacher_fast_path", False)
    )
    require_teacher_cache_contract = bool(
        train_cfg.get(
            "teacher_logits_cache_require_contract",
            cached_teacher_fast_path,
        )
    )
    teacher_cache_contract_kwargs = {
        "require_contract": require_teacher_cache_contract,
        "expected_class_to_idx": class_to_idx,
        "expected_manifest_sha256": data_cfg.get("train_manifest_sha256"),
        "expected_image_size": int(data_cfg.get("image_size", 224)),
        "expected_resize_mode": str(
            data_cfg.get("train_resize_mode", "letterbox")
        ),
        "expected_augmentation": bool(train_cfg.get("augmentation", True)),
    }
    multi_teacher_specs = train_cfg.get("teacher_logits_caches")
    if multi_teacher_specs is not None:
        _, resolved_student_hw = _resolve_data_input_geometry(data_cfg)
        legacy_student_size = int(data_cfg.get("image_size", 224))
        if resolved_student_hw != (legacy_student_size, legacy_student_size):
            raise ValueError(
                "rectangular students currently require a single pinned "
                "teacher cache; multi-teacher provenance still uses the "
                "legacy scalar student-view contract"
            )
        if train_cfg.get("teacher_logits_cache") or train_cfg.get(
            "expert_teacher_logits_cache"
        ):
            raise ValueError(
                "train.teacher_logits_caches cannot be combined with legacy "
                "teacher_logits_cache/expert_teacher_logits_cache"
            )
        if not require_teacher_cache_contract:
            raise ValueError(
                "multi-teacher caches require teacher_logits_cache_require_contract=true"
            )
        ensemble_cfg = train_cfg.get("teacher_logits_ensemble", {})
        if not isinstance(ensemble_cfg, Mapping):
            raise TypeError("train.teacher_logits_ensemble must be a mapping")
        train_manifest_sha256 = data_cfg.get("train_manifest_sha256")
        if not train_manifest_sha256:
            raise ValueError(
                "multi-teacher cache loading requires data.train_manifest_sha256"
            )
        anchor_teacher_logit_cache = load_teacher_logit_ensemble_cache(
            multi_teacher_specs,
            require_contract=True,
            expected_class_to_idx=class_to_idx,
            expected_manifest_sha256=str(train_manifest_sha256),
            student_image_size=int(data_cfg.get("image_size", 224)),
            student_resize_mode=str(
                data_cfg.get("train_resize_mode", "letterbox")
            ),
            student_augmentation=bool(train_cfg.get("augmentation", True)),
            expected_split_roles=("train", "b0500"),
            method=str(
                ensemble_cfg.get("method", "equal_log_probability_mean")
            ),
            expected_ensemble_provenance_sha256=ensemble_cfg.get(
                "provenance_sha256"
            ),
            require_pinned_hashes=bool(
                ensemble_cfg.get("require_pinned_hashes", True)
            ),
            allow_deterministic_cross_view=bool(
                ensemble_cfg.get("allow_deterministic_cross_view", False)
            ),
        )
        expert_teacher_logit_cache = None
    else:
        relation_weight = float(
            cfg.get("loss", {}).get(
                "cached_teacher_relation_distill_weight",
                0.0,
            )
        )
        relation_representation_key = (
            str(
                cfg.get("loss", {}).get(
                    "cached_teacher_relation_distill_representation_key",
                    "backbone_embedding",
                )
            ).strip()
            if relation_weight > 0.0
            else None
        )
        coordinate_weight = float(
            cfg.get("loss", {}).get(
                "cached_teacher_coordinate_distill_weight",
                0.0,
            )
        )
        coordinate_representation_key = (
            str(
                cfg.get("loss", {}).get(
                    "cached_teacher_coordinate_distill_representation_key",
                    "backbone_embedding",
                )
            ).strip()
            if coordinate_weight > 0.0
            else None
        )
        requested_representation_keys = {
            key
            for key in (
                relation_representation_key,
                coordinate_representation_key,
            )
            if key
        }
        if len(requested_representation_keys) > 1:
            raise ValueError(
                "a single teacher cache can load only one representation per "
                "run; coordinate and relation representation keys must match"
            )
        requested_representation_key = next(
            iter(requested_representation_keys),
            None,
        )
        raw_single_cache_contract = train_cfg.get(
            "teacher_logits_cache_contract",
            {},
        )
        if raw_single_cache_contract is None:
            raw_single_cache_contract = {}
        if not isinstance(raw_single_cache_contract, Mapping):
            raise TypeError(
                "train.teacher_logits_cache_contract must be a mapping"
            )
        single_cache_contract = dict(raw_single_cache_contract)
        single_cache_contract_present = bool(single_cache_contract)
        anchor_teacher_logit_cache = load_teacher_logit_cache(
            train_cfg.get("teacher_logits_cache"),
            require_contract=require_teacher_cache_contract,
            expected_class_to_idx=class_to_idx,
            expected_manifest_sha256=single_cache_contract.get(
                "manifest_sha256",
                data_cfg.get("train_manifest_sha256"),
            ),
            expected_image_size=int(
                single_cache_contract.get(
                    "image_size",
                    data_cfg.get("image_size", 224),
                )
            ),
            expected_resize_mode=str(
                single_cache_contract.get(
                    "resize_mode",
                    data_cfg.get("train_resize_mode", "letterbox"),
                )
            ),
            expected_augmentation=(
                False
                if single_cache_contract_present
                else bool(train_cfg.get("augmentation", True))
            ),
            expected_cache_sha256=single_cache_contract.get("cache_sha256"),
            expected_provenance_sha256=single_cache_contract.get(
                "provenance_sha256"
            ),
            expected_checkpoint_sha256=single_cache_contract.get(
                "checkpoint_sha256"
            ),
            expected_split_roles=(
                ("train", "b0500")
                if single_cache_contract_present
                else None
            ),
            logits_key=str(single_cache_contract.get("logits_key", "logits")),
            representation_key=requested_representation_key,
            cache_name="anchor",
        )
        _audit_single_teacher_student_view(
            anchor_teacher_logit_cache,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            cache_contract=single_cache_contract,
        )
        raw_expert_cache_contract = train_cfg.get(
            "expert_teacher_logits_cache_contract",
            {},
        )
        if raw_expert_cache_contract is None:
            raw_expert_cache_contract = {}
        if not isinstance(raw_expert_cache_contract, Mapping):
            raise TypeError(
                "train.expert_teacher_logits_cache_contract must be a mapping"
            )
        expert_cache_contract = dict(raw_expert_cache_contract)
        if expert_cache_contract:
            expert_teacher_logit_cache = load_teacher_logit_cache(
                train_cfg.get("expert_teacher_logits_cache"),
                require_contract=True,
                expected_class_to_idx=class_to_idx,
                expected_manifest_sha256=expert_cache_contract.get(
                    "manifest_sha256",
                    data_cfg.get("train_manifest_sha256"),
                ),
                expected_image_size=int(expert_cache_contract["image_size"]),
                expected_resize_mode=str(expert_cache_contract["resize_mode"]),
                expected_augmentation=bool(
                    expert_cache_contract.get("augmentation", False)
                ),
                expected_cache_sha256=expert_cache_contract.get("cache_sha256"),
                expected_checkpoint_sha256=expert_cache_contract.get(
                    "checkpoint_sha256"
                ),
                expected_split_roles=("train",),
                logits_key=str(expert_cache_contract.get("logits_key", "logits")),
                cache_name="expert",
                allow_path_aligned_cross_view_expert=bool(
                    expert_cache_contract.get(
                        "allow_path_aligned_cross_view_expert",
                        False,
                    )
                ),
            )
        else:
            expert_teacher_logit_cache = load_teacher_logit_cache(
                train_cfg.get("expert_teacher_logits_cache"),
                **teacher_cache_contract_kwargs,
            )
    eval_cfg = cfg.get("eval", {})
    if not isinstance(eval_cfg, Mapping):
        raise TypeError("eval configuration must be a mapping")
    raw_eval_cache_contract = eval_cfg.get("teacher_logits_cache_contract", {})
    if raw_eval_cache_contract is None:
        raw_eval_cache_contract = {}
    if not isinstance(raw_eval_cache_contract, Mapping):
        raise TypeError("eval.teacher_logits_cache_contract must be a mapping")
    eval_cache_contract = dict(raw_eval_cache_contract)
    eval_cache_path = eval_cfg.get("teacher_logits_cache")
    if eval_cache_path and not eval_cache_contract:
        raise ValueError(
            "eval.teacher_logits_cache requires a fully pinned "
            "eval.teacher_logits_cache_contract"
        )
    facet_validation_teacher_logit_cache = load_teacher_logit_cache(
        eval_cache_path,
        require_contract=bool(eval_cache_path),
        expected_class_to_idx=class_to_idx,
        expected_manifest_sha256=eval_cache_contract.get(
            "manifest_sha256",
            data_cfg.get("val_manifest_sha256"),
        ),
        expected_image_size=(
            int(eval_cache_contract["image_size"])
            if "image_size" in eval_cache_contract
            else None
        ),
        expected_resize_mode=eval_cache_contract.get("resize_mode"),
        expected_augmentation=False if eval_cache_path else None,
        expected_cache_sha256=eval_cache_contract.get("cache_sha256"),
        expected_provenance_sha256=eval_cache_contract.get(
            "provenance_sha256"
        ),
        expected_checkpoint_sha256=eval_cache_contract.get(
            "checkpoint_sha256"
        ),
        expected_split_roles=("val", "validation"),
        logits_key=str(eval_cache_contract.get("logits_key", "logits")),
        cache_name="facet_validation",
    )
    use_teacher_cache_fallback = bool(train_cfg.get("teacher_cache_online_fallback", False))
    teacher_model = None if anchor_teacher_logit_cache is not None and not use_teacher_cache_fallback else build_anchor_teacher(cfg, class_to_idx, device)
    expert_teacher_model = (
        None
        if expert_teacher_logit_cache is not None and not use_teacher_cache_fallback
        else build_specialist_teacher(cfg, class_to_idx, device)
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    base_lr = float(train_cfg.get("lr", 3.5e-5))
    fused_adamw_requested = bool(train_cfg.get("fused_adamw", False))
    fused_adamw = fused_adamw_requested
    fused_adamw_fallback_reason: str | None = None
    if fused_adamw and device.type != "cuda":
        print("WARNING: train.fused_adamw requires CUDA; using standard AdamW")
        fused_adamw = False
        fused_adamw_fallback_reason = "non_cuda_device"
    # -------- 第 5 阶段：建立 AdamW 与 AMP scaler --------
    # 对精确续训而言，下面创建的初始状态会立刻被 checkpoint 状态完整覆盖。
    optimizer = torch.optim.AdamW(
        trainable,
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.003)),
        fused=fused_adamw,
    )
    pcgrad_enabled = bool(
        cfg.get("loss", {}).get(
            "rscd_pcgrad_enabled",
            train_cfg.get("rscd_pcgrad_enabled", False),
        )
    )
    requested_amp_dtype = _amp_autocast_dtype(train_cfg, device)
    amp_dtype = None if pcgrad_enabled else requested_amp_dtype
    eval_amp_dtype = _eval_autocast_dtype(train_cfg, device)
    runtime_audit["fused_adamw_requested"] = fused_adamw_requested
    runtime_audit["fused_adamw_effective"] = fused_adamw
    runtime_audit["fused_adamw_fallback_reason"] = fused_adamw_fallback_reason
    runtime_audit["pcgrad_enabled"] = pcgrad_enabled
    runtime_audit["linear_ce_fast_path"] = bool(
        train_cfg.get("linear_ce_fast_path", False)
    )
    runtime_audit["self_anchor_fast_path"] = bool(
        train_cfg.get("self_anchor_fast_path", False)
    )
    runtime_audit["cached_teacher_fast_path"] = bool(
        train_cfg.get("cached_teacher_fast_path", False)
    )
    runtime_audit["teacher_logits_cache_require_contract"] = bool(
        require_teacher_cache_contract
    )
    runtime_audit["teacher_logits_cache_audit"] = (
        copy.deepcopy(anchor_teacher_logit_cache.audit)
        if isinstance(anchor_teacher_logit_cache, TeacherLogitCache)
        else None
    )
    runtime_audit["expert_teacher_logits_cache_audit"] = (
        copy.deepcopy(expert_teacher_logit_cache.audit)
        if isinstance(expert_teacher_logit_cache, TeacherLogitCache)
        else None
    )
    runtime_audit["eval_teacher_logits_cache_audit"] = (
        copy.deepcopy(facet_validation_teacher_logit_cache.audit)
        if isinstance(facet_validation_teacher_logit_cache, TeacherLogitCache)
        else None
    )
    runtime_audit["train_amp_requested_dtype"] = (
        str(requested_amp_dtype) if requested_amp_dtype is not None else None
    )
    runtime_audit["train_amp_dtype"] = (
        str(amp_dtype) if amp_dtype is not None else None
    )
    runtime_audit["train_amp_disabled_reason"] = (
        "pcgrad_requires_fp32" if pcgrad_enabled and requested_amp_dtype is not None else None
    )
    runtime_audit["eval_amp_dtype"] = (
        str(eval_amp_dtype) if eval_amp_dtype is not None else None
    )
    (
        skip_nonfinite_grad_steps,
        max_nonfinite_grad_skips,
    ) = _nonfinite_gradient_recovery_settings(train_cfg)
    runtime_audit["skip_nonfinite_grad_steps"] = bool(
        skip_nonfinite_grad_steps
    )
    runtime_audit["max_nonfinite_grad_skips"] = int(
        max_nonfinite_grad_skips
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
        and bool(train_cfg.get("amp", True))
        and amp_dtype == torch.float16,
    )
    if step_resume_state is not None:
        # 恢复顺序：optimizer → scaler → RNG。RNG 必须在下一次取 batch 前恢复。
        if "optimizer" in step_resume_state:
            optimizer.load_state_dict(step_resume_state["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if "scaler" in step_resume_state:
            scaler_state = step_resume_state["scaler"]
            saved_scaler_enabled = step_resume_state.get("scaler_enabled")
            if saved_scaler_enabled is None:
                # Legacy GradScaler state is empty when disabled and populated
                # when enabled.  Future checkpoints save the boolean directly.
                saved_scaler_enabled = bool(scaler_state)
            if bool(saved_scaler_enabled) != bool(scaler.is_enabled()):
                raise RuntimeError(
                    "step checkpoint GradScaler mode does not match the "
                    "current effective AMP dtype; refusing a non-exact resume: "
                    f"checkpoint_enabled={bool(saved_scaler_enabled)} "
                    f"current_enabled={bool(scaler.is_enabled())}"
                )
            scaler.load_state_dict(scaler_state)
        _restore_rng_state(step_resume_state.get("rng_state"))
    fused_group_values = [
        bool(group.get("fused", False)) for group in optimizer.param_groups
    ]
    runtime_audit["fused_adamw_group_values"] = fused_group_values
    runtime_audit["fused_adamw_effective"] = bool(fused_group_values) and all(
        fused_group_values
    )
    runtime_audit["fused_adamw_restored_from_checkpoint"] = bool(
        step_resume_state is not None and "optimizer" in step_resume_state
    )
    if (
        runtime_audit["fused_adamw_restored_from_checkpoint"]
        and runtime_audit["fused_adamw_effective"] != fused_adamw_requested
    ):
        runtime_audit["fused_adamw_fallback_reason"] = (
            "checkpoint_param_group_is_authoritative_for_exact_resume"
        )
    runtime_audit["grad_scaler_enabled"] = bool(scaler.is_enabled())
    (out_dir / "runtime_audit.json").write_text(
        json.dumps(runtime_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    provenance_path = out_dir / "run_provenance.json"
    try:
        run_provenance = build_run_provenance(
            config_path=config_path,
            resolved_config_path=out_dir / "config_resolved.yaml",
            repo_root=Path(__file__).resolve().parents[2],
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            runtime_audit=runtime_audit,
        )
        write_run_provenance(run_provenance, provenance_path)
    except Exception as exc:  # Provenance must never change training semantics.
        run_provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "status": "degraded",
            "seed": int(cfg.get("seed", 79)),
            "error": f"{type(exc).__name__}: {exc}",
        }
        run_provenance["provenance_sha256"] = canonical_sha256(run_provenance)
        try:
            write_run_provenance(run_provenance, provenance_path)
        except OSError as write_error:
            print(
                "WARNING: could not persist run provenance; training will "
                "continue unchanged "
                f"({type(write_error).__name__}: {write_error})"
            )
    checkpoint_score_weights = train_cfg.get("checkpoint_score_weights")
    use_weighted_checkpoint_score = isinstance(checkpoint_score_weights, dict)

    def validation_key(summary: dict[str, Any]) -> float | tuple[float, float]:
        if use_weighted_checkpoint_score:
            return weighted_validation_score(summary, checkpoint_score_weights)
        return (float(summary["macro_f1"]), float(summary["top1"]))

    def checkpoint_state(epoch: int, val_summary: dict[str, Any]) -> dict[str, Any]:
        evaluation_model = ema.module if ema is not None else model
        state: dict[str, Any] = {
            "model": evaluation_model.state_dict(),
            "epoch": int(epoch),
            "class_to_idx": class_to_idx,
            "config": cfg,
            "load_audit": dict(load_audit),
            "provenance": copy.deepcopy(run_provenance),
            "val_summary": val_summary,
        }
        if ema is not None:
            state["training_model"] = model.state_dict()
            state["ema"] = ema.state_dict()
        return state

    best_key: float | tuple[float, float]
    best_key = float("-inf") if use_weighted_checkpoint_score else (-1.0, -1.0)
    history_path = out_dir / "history.json"
    history: list[dict[str, Any]] = []
    if step_resume_state is not None and history_path.exists():
        try:
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(loaded_history, list):
                history = [
                    item
                    for item in loaded_history
                    if isinstance(item, dict)
                    and int(item.get("epoch", -1)) < start_epoch
                ]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                "WARNING: could not restore prior history.json; "
                f"continuing with a fresh history list ({type(exc).__name__}: {exc})"
            )
    best_checkpoint_path = out_dir / "best_checkpoint.pth"
    if step_resume_state is not None and best_checkpoint_path.exists():
        try:
            prior_best = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            prior_summary = prior_best.get("val_summary")
            if isinstance(prior_summary, dict):
                best_key = validation_key(prior_summary)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(
                "WARNING: could not recover prior best checkpoint score; "
                f"the next validation will re-establish it ({type(exc).__name__}: {exc})"
            )
    # 监控只读取已经产生的指标，不参与梯度或模型计算。已有 1--49 轮会在首次
    # 运行时自动补入 metrics.csv 与 TensorBoard，方便跨电脑后查看完整曲线。
    monitor = TrainingMonitor(out_dir, history)
    if bool(train_cfg.get("evaluate_initial", False)) and step_resume_state is None:
        print("Evaluating initial checkpoint before fine-tuning")
        evaluation_model = ema.module if ema is not None else model
        # Persist the exact epoch-0 predictions and state before any optimizer
        # step.  Mechanism studies (for example METER projective versus its
        # matched controls) need to verify per-image initialization equality;
        # a metric summary alone is not sufficient evidence.
        val_metrics = evaluate(
            evaluation_model,
            val_loader,
            device,
            idx_to_class,
            save_predictions_path=out_dir / "predictions_val_epoch0.csv",
            save_logits_path=out_dir / "logits_val_epoch0_fp32.npy",
            amp_dtype=eval_amp_dtype,
            facet_teacher_logit_cache=facet_validation_teacher_logit_cache,
            facet_teacher_temperature=float(
                cfg.get("loss", {}).get(
                    "facet_joint_distill_temperature",
                    2.0,
                )
            ),
        )
        val_summary = val_metrics["summary"]
        initial_key = validation_key(val_summary)
        print(f"  initial val top1={val_summary['top1']:.4f} macro_f1={val_summary['macro_f1']:.4f} wcs={val_summary['water_concrete_slight_f1']:.4f}")
        initial_record = {
                "epoch": 0,
                "train": {},
                "val": val_summary,
                "checkpoint_score": initial_key,
                "provenance_sha256": run_provenance["provenance_sha256"],
            }
        history.append(initial_record)
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        state = checkpoint_state(0, val_summary)
        torch.save(state, out_dir / "initial_checkpoint.pth")
        torch.save(state, out_dir / "initial.pt")
        torch.save(state, out_dir / "best_checkpoint.pth")
        torch.save(state, out_dir / "best.pt")
        best_key = initial_key
        monitor.log_epoch(initial_record, is_best=True)
        print(f"  saved initial best: {out_dir / 'best_checkpoint.pth'}")
    previous_sampling_mode = _balanced_sampling_active(train_cfg, start_epoch)
    # -------- 第 6 阶段：主训练循环 --------
    # start_epoch=50 时，train_one_epoch 会根据 _resume_start_step 跳过前 2500 step。
    for epoch in range(start_epoch, configured_epochs + 1):
        cfg["train"]["_current_epoch"] = int(epoch)
        current_sampling_mode = _balanced_sampling_active(train_cfg, epoch)
        if epoch > start_epoch:
            cfg["train"]["_resume_start_step"] = 0
            cfg["train"]["_resume_epoch_training_complete"] = False
            cfg["train"]["_resume_train_partial"] = {}
            can_reuse_loader = (
                current_sampling_mode == previous_sampling_mode
                and _set_train_loader_epoch(train_loader, epoch)
            )
            if can_reuse_loader:
                print(
                    "Reusing persistent training DataLoader workers: "
                    f"epoch={epoch} balanced_sampling={current_sampling_mode}"
                )
            else:
                train_loader, _, _ = build_loaders(
                    cfg,
                    class_to_idx,
                    include_test=False,
                )
            previous_sampling_mode = current_sampling_mode
        if "warmup_epochs" in train_cfg or "min_lr" in train_cfg:
            current_lr = set_warmup_cosine_lr(
                optimizer,
                epoch=epoch,
                total_epochs=int(cfg["train"].get("epochs", 1)),
                warmup_epochs=int(train_cfg.get("warmup_epochs", 0)),
                base_lr=base_lr,
                min_lr=float(train_cfg.get("min_lr", base_lr)),
            )
        else:
            current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch}/{cfg['train'].get('epochs', 1)} "
            f"lr={current_lr:.8g} balanced_sampling={current_sampling_mode}"
        )
        if bool(cfg["train"].get("_resume_epoch_training_complete", False)) and epoch == start_epoch and step_resume_state is not None:
            train_partial = dict(step_resume_state.get("train_partial", {}) or {})
            train_metrics = {
                "loss": float(train_partial.get("loss", 0.0) or 0.0),
                "top1": float(train_partial.get("top1", 0.0) or 0.0),
            }
            partial_aux_sum = train_partial.get("aux_log_sum", {})
            partial_aux_count = max(
                int(train_partial.get("aux_log_count", 0) or 0),
                1,
            )
            if isinstance(partial_aux_sum, dict):
                train_metrics.update(
                    {
                        str(key): float(value) / partial_aux_count
                        for key, value in partial_aux_sum.items()
                    }
                )
            print(
                "  skipped training epoch from completed step checkpoint "
                f"step={int(step_resume_state.get('step', 0) or 0)}/"
                f"{int(step_resume_state.get('total_steps', 0) or 0)}"
            )
        else:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                scaler,
                teacher_model=teacher_model,
                expert_teacher_model=expert_teacher_model,
                anchor_teacher_logit_cache=anchor_teacher_logit_cache,
                expert_teacher_logit_cache=expert_teacher_logit_cache,
                teacher_cache_strict=bool(train_cfg.get("teacher_logits_cache_strict", False)),
                idx_to_class=idx_to_class,
                out_dir=out_dir,
                epoch=epoch,
                class_to_idx=class_to_idx,
                ema=ema,
                run_provenance=run_provenance,
                progress_callback=monitor.log_train_progress,
            )
        train_metrics["lr"] = float(current_lr)
        evaluation_model = ema.module if ema is not None else model
        if bool(train_cfg.get("release_train_workers_before_eval", False)) or _env_flag_enabled(
            "DREL_RELEASE_TRAIN_WORKERS_BEFORE_EVAL"
        ):
            _release_dataloader_workers(train_loader)
            gc.collect()
        val_metrics = evaluate(
            evaluation_model,
            val_loader,
            device,
            idx_to_class,
            amp_dtype=eval_amp_dtype,
            facet_teacher_logit_cache=facet_validation_teacher_logit_cache,
            facet_teacher_temperature=float(
                cfg.get("loss", {}).get(
                    "facet_joint_distill_temperature",
                    2.0,
                )
            ),
        )
        val_summary = val_metrics["summary"]
        key = validation_key(val_summary)
        print(f"  train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.4f}")
        print(
            f"  val top1={val_summary['top1']:.4f} "
            f"macro_f1={val_summary['macro_f1']:.4f} "
            f"bottom5={val_summary['bottom5_mean_f1']:.4f} "
            f"wcs={val_summary['water_concrete_slight_f1']:.4f} "
            f"checkpoint_score={key}"
        )
        # 一个 epoch 完整结束且验证完成后，才向 history 追加一行。
        epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_summary,
                "checkpoint_score": key,
                "provenance_sha256": run_provenance["provenance_sha256"],
            }
        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        state = checkpoint_state(epoch, val_summary)
        # last 保存最新完整轮；best 只在加权验证分数提高时替换。
        # epoch 内更细的恢复状态由 train_one_epoch 写入 last_step_checkpoint.pth。
        torch.save(state, out_dir / "last_checkpoint.pth")
        torch.save(state, out_dir / "last.pt")
        is_best = key > best_key
        if is_best:
            best_key = key
            torch.save(state, out_dir / "best_checkpoint.pth")
            torch.save(state, out_dir / "best.pt")
            print(f"  saved best: {out_dir / 'best_checkpoint.pth'}")
        monitor.log_epoch(epoch_record, is_best=is_best)
        if epoch >= stop_after_epoch:
            print(
                "Stopped at the frozen development gate: "
                f"epoch={epoch}/{configured_epochs}."
            )
            break
    if not run_test_after_train:
        best_state = torch.load(
            out_dir / "best_checkpoint.pth",
            map_location="cpu",
            weights_only=False,
        )
        print(
            "Training complete. Test evaluation was intentionally skipped; "
            "run test.py only after the architecture and protocol are frozen."
        )
        print(json.dumps(best_state.get("val_summary", {}), indent=2, ensure_ascii=False))
        monitor.close()
        return
    state = torch.load(out_dir / "best_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    if test_loader is None:
        raise RuntimeError("test loader was not constructed for post-training evaluation")
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        idx_to_class,
        save_predictions_path=out_dir / "predictions_test.csv",
        amp_dtype=eval_amp_dtype,
    )
    write_outputs(out_dir, test_metrics, idx_to_class, split="test")
    print(json.dumps(test_metrics["summary"], indent=2, ensure_ascii=False))
    monitor.close()


def _validate_eval_checkpoint_contract(
    cfg: Mapping[str, Any],
    state: Mapping[str, Any],
    class_to_idx: Mapping[str, int],
    checkpoint: Path,
) -> dict[str, Any]:
    """Fail closed when an eval config can change checkpoint semantics.

    Some matched-control backbones intentionally share an identical state
    schema while selecting different non-state forward rules in the model
    config.  A strict ``load_state_dict`` cannot detect a CETA/generic swap, so
    opt-in experiments bind evaluation to the model and data config embedded
    in the checkpoint as well as to its epoch and class map.
    """

    eval_cfg = cfg.get("eval", {})
    require_model = bool(
        eval_cfg.get("require_checkpoint_model_config_match", False)
    )
    require_data = bool(
        eval_cfg.get("require_checkpoint_data_config_match", False)
    )
    required_epoch = eval_cfg.get("require_checkpoint_epoch")
    if not require_model and not require_data and required_epoch is None:
        return {"enabled": False}

    saved_cfg = state.get("config")
    if not isinstance(saved_cfg, Mapping):
        raise RuntimeError(
            f"evaluation checkpoint is missing its config contract: {checkpoint}"
        )
    if require_model and saved_cfg.get("model") != cfg.get("model"):
        saved_backbone = (
            saved_cfg.get("model", {}).get("backbone")
            if isinstance(saved_cfg.get("model"), Mapping)
            else None
        )
        current_backbone = (
            cfg.get("model", {}).get("backbone")
            if isinstance(cfg.get("model"), Mapping)
            else None
        )
        raise RuntimeError(
            "evaluation checkpoint model config mismatch; refusing a "
            "shape-compatible semantic arm swap: "
            f"checkpoint_backbone={saved_backbone!r} "
            f"current_backbone={current_backbone!r}"
        )
    if require_data and saved_cfg.get("data") != cfg.get("data"):
        raise RuntimeError(
            "evaluation checkpoint data config mismatch; refusing to change "
            "the manifest or preprocessing contract"
        )
    if required_epoch is not None:
        actual_epoch = state.get("epoch")
        if actual_epoch is None or int(actual_epoch) != int(required_epoch):
            raise RuntimeError(
                "evaluation checkpoint epoch mismatch: "
                f"expected={int(required_epoch)} actual={actual_epoch!r}"
            )
    saved_class_to_idx = state.get("class_to_idx")
    if not isinstance(saved_class_to_idx, Mapping):
        raise RuntimeError("evaluation checkpoint is missing class_to_idx")
    normalized_saved = {
        str(name): int(index) for name, index in saved_class_to_idx.items()
    }
    normalized_current = {
        str(name): int(index) for name, index in class_to_idx.items()
    }
    if normalized_saved != normalized_current:
        raise RuntimeError("evaluation checkpoint class_to_idx mismatch")
    return {
        "enabled": True,
        "checkpoint": str(Path(checkpoint).resolve()),
        "model_config_match": require_model,
        "data_config_match": require_data,
        "required_epoch": int(required_epoch) if required_epoch is not None else None,
        "actual_epoch": int(state["epoch"]) if state.get("epoch") is not None else None,
        "class_to_idx_match": True,
    }


def run_eval(
    config_path: Path,
    checkpoint: Path,
    split: str = "test",
    *,
    seed_override: int | None = None,
    output_dir_override: Path | None = None,
    logit_patch_rules_path: Path | None = None,
) -> None:
    cfg = load_config(config_path)
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    if output_dir_override is not None:
        cfg["output_dir"] = str(output_dir_override)
    eval_cfg = cfg.get("eval", {})
    if logit_patch_rules_path is None and eval_cfg.get("logit_patch_rules_path"):
        logit_patch_rules_path = Path(str(eval_cfg["logit_patch_rules_path"]))
    logit_patch_rules = load_pareto_safe_logit_patch_rules(logit_patch_rules_path)
    out_dir = Path(cfg["output_dir"]) / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = cfg["data"]
    manifests = [
        _runtime_path(data["train_manifest"]),
        _runtime_path(data["val_manifest"]),
    ]
    if split == "test":
        manifests.append(_runtime_path(data["test_manifest"]))
    class_to_idx = build_class_map(manifests)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    (out_dir / "label_factor_sanity.txt").write_text(sanity_summary(class_to_idx), encoding="utf-8")
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _, val_loader, test_loader = build_loaders(
        cfg,
        class_to_idx,
        include_test=split == "test",
    )
    if split == "val":
        loader = val_loader
    else:
        if test_loader is None:
            raise RuntimeError("test evaluation requires a test loader")
        loader = test_loader
    device = resolve_device()
    runtime_audit = _configure_cuda_runtime(cfg.get("train", {}), device)
    eval_amp_dtype = _eval_autocast_dtype(cfg.get("train", {}), device)
    runtime_audit["eval_amp_dtype"] = (
        str(eval_amp_dtype) if eval_amp_dtype is not None else None
    )
    (out_dir / "runtime_audit.json").write_text(
        json.dumps(runtime_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    model = build_model(cfg, class_to_idx).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_contract = _validate_eval_checkpoint_contract(
        cfg,
        state,
        class_to_idx,
        checkpoint,
    )
    (out_dir / "checkpoint_contract.json").write_text(
        json.dumps(checkpoint_contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    model.load_state_dict(state["model"], strict=True)
    metrics = evaluate(
        model,
        loader,
        device,
        idx_to_class,
        save_predictions_path=out_dir / f"predictions_{split}.csv",
        logit_patch_rules=logit_patch_rules,
        amp_dtype=eval_amp_dtype,
    )
    write_outputs(out_dir, metrics, idx_to_class, split=split)
    materialize_eval_checkpoint_aliases(out_dir, checkpoint, split=split)
    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))
