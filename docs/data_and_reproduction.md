# Data and Reproduction

## Dataset

The current verified result uses RSCD full train/validation/test splits:

| split | images |
|---|---:|
| train | 958,941 |
| validation | 19,860 |
| test | 49,500 |

The dataset itself is not redistributed in this repository. Download RSCD from its official source and prepare the local paths.

## Manifest Format

The code reads CSV manifests. Required columns:

```text
image_path,split,dataset,class_label,domain_id,friction_label,material_label,unevenness_label,wetness_label,snow_label,risk_label,mu_low,mu_high
```

Important columns:

- `image_path`: absolute or relative path to an image
- `split`: train, val, or test
- `dataset`: usually `rscd`
- `class_label`: one of the 27 RSCD classes
- `friction_label`: dry, wet, water, fresh_snow, melted_snow, ice
- `material_label`: asphalt, concrete, mud, gravel, none
- `unevenness_label`: smooth, slight, severe, none
- `mu_low`, `mu_high`: weak visual friction-risk interval derived from road-state labels

## Generate Manifests

Create a local path file:

```bash
cp configs/data/local_paths.example.yaml configs/data/local_paths.yaml
```

Edit `configs/data/local_paths.yaml`, then run:

```bash
python scripts/build_manifests.py --config configs/data/local_paths.yaml --out-dir data/manifests_full
```

## Train Current Public Config

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

## Test

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint outputs/current_best_s7/best_checkpoint.pth
```

## Reproduction Note

The exact verified historical S7 run used warm-start checkpoints produced by earlier screening runs. The selected final, parent, and teacher artifacts are now tracked with Git LFS under `checkpoints/`; run `git lfs pull` before evaluation or continuation training. Their sizes and SHA256 hashes are listed in `results/s7_lineage/checkpoint_manifest.json`.

The portable public config already points to the uploaded artifacts:

```yaml
train:
  resume_from: checkpoints/c3_farnet_errorgate_paircal_screen_20260703/best_checkpoint.pth
  teacher_checkpoint: checkpoints/screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor/best.pt
```

This reproduces the historical continuation recipe, not a from-scratch training claim. The full ancestry and the distinction between the self-contained S7 checkpoint and the parent-plus-router inference variant are documented in `docs/s7_training_lineage.md`.
