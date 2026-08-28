from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from drel_qrfme.rscd_label_factors import fixed_rscd_class_map
from drel_qrfme.training_engine import build_model, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/drel_qrfme/rscd_full_train_seed097.yaml"
CHECKPOINT = ROOT / "checkpoints/drel_qrfme_epoch097/best_checkpoint.pth"
EXPECTED_SHA256 = "72c2c2cd34545dcc1e22d5c89d43d1c79b3278a3edd7bc5857e0ab0a493dff55"


def _model() -> torch.nn.Module:
    cfg = load_config(CONFIG)
    return build_model(cfg, fixed_rscd_class_map())


def test_frozen_architecture_contract() -> None:
    model = _model()
    assert model.architecture_version == "drel_rt_rfme_v1"
    assert model.variant == "rfme"
    assert sum(parameter.numel() for parameter in model.parameters()) == 26_145_426
    assert float(model.max_write_ratio) == pytest.approx(0.05)


def test_forward_contract_and_budget() -> None:
    model = _model().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224), return_aux=True)
    assert output["logits"].shape == (1, 27)
    assert torch.isfinite(output["logits"]).all()
    ratios = output["drel_rt_write_ratio"]
    budgets = output["drel_rt_write_budgets"]
    assert ratios.shape == (1, 4)
    assert torch.all(ratios <= budgets.unsqueeze(0) + 1.0e-6)


def test_released_checkpoint_is_strictly_compatible() -> None:
    prefix = CHECKPOINT.read_bytes()[:200]
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        pytest.skip("Git LFS checkpoint content was not fetched")
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert int(payload["epoch"]) == 97
    model = _model()
    incompatible = model.load_state_dict(payload["model"], strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
