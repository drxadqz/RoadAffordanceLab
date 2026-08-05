"""Minimal DREL construction, inference, and one optimization step."""

from __future__ import annotations

import torch

from friction_affordance.models.drel import build_drel_component_full


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_drel_component_full(
        num_classes=27,
        head_init_seed=970027,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # The frozen experiment contract stores the tensor as B x 3 x 360 x 240.
    image = torch.randn(2, 3, 360, 240, device=device)
    target = torch.tensor([0, 26], device=device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        details = model(image[:1], return_aux=True)

    print(f"device={device}")
    print(f"parameters={parameter_count:,}")
    print(f"training_loss={loss.item():.6f}")
    print(f"logits_shape={tuple(details['logits'].shape)}")
    print(f"stage_write_rms={details['drel']['drel_write_rms'].cpu().tolist()}")


if __name__ == "__main__":
    main()
