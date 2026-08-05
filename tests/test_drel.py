from __future__ import annotations

import torch

from friction_affordance.models.drel import (
    DRELBackbone,
    MatchedResponseConv2d,
    build_drel_component_full,
)


def test_frozen_parameter_counts_and_schema() -> None:
    backbone = DRELBackbone()
    model = build_drel_component_full(num_classes=27, head_init_seed=970027)

    assert sum(parameter.numel() for parameter in backbone.parameters()) == 2_751_694
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_760_361
    state = backbone.state_dict()
    assert state["_drel_ledger_mode_code"].item() == 1
    assert state["_drel_schema_version"].item() == 1
    assert state["_drel_component_study_code"].item() == 0
    assert state["_drel_component_study_schema"].item() == 1


def test_matched_response_constraints_hold() -> None:
    bank = MatchedResponseConv2d(3, 16, 7, orientations=4)

    assert bank.zero_dc_error().item() < 2.0e-6
    assert bank.unit_norm_error().item() < 2.0e-6
    assert bank.pair_orthogonality_error().item() < 2.0e-6


def test_forward_auxiliary_contract() -> None:
    torch.manual_seed(7)
    model = build_drel_component_full(num_classes=27, head_init_seed=970027)
    model.eval()
    image = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output = model(image, return_aux=True)

    assert output["logits"].shape == (1, 27)
    assert output["embedding"].shape == (1, 320)
    assert output["drel"]["drel_write_rms"].shape == (1, 4)
    assert output["drel"]["drel_mean_reliability"].shape == (1, 4)
    assert tuple(output["stage_feature_maps"]) == ("early", "mid", "late", "final")
    assert torch.isfinite(output["logits"]).all()


def test_checkpoint_semantics_fail_closed() -> None:
    source = DRELBackbone()
    state = source.state_dict()
    state["_drel_ledger_mode_code"] = torch.tensor(2, dtype=torch.int64)

    target = DRELBackbone()
    try:
        target.load_state_dict(state, strict=False)
    except RuntimeError as error:
        assert "semantic buffer" in str(error)
    else:
        raise AssertionError("a mismatched DREL ledger mode must not load")

    assert target._drel_ledger_mode_code.item() == 1


def test_backward_is_finite() -> None:
    torch.manual_seed(11)
    model = build_drel_component_full(num_classes=27, head_init_seed=970027)
    model.train()
    image = torch.randn(2, 3, 64, 64)
    target = torch.tensor([0, 26])

    logits = model(image)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
