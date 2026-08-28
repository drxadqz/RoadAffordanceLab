"""论文开源版训练入口。

推荐普通用户运行 ``scripts/run_training.py``；本文件提供符合常见开源仓库习惯的
低层入口，适合已经正确设置 DREL_PATH_REMAP_JSON 的高级用户。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from drel_qrfme.training_engine import run_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DREL-QRFME on RSCD.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/drel_qrfme/rscd_full_train_seed097.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="RSCD root containing train/ and vali_20k/",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/seed097")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "train").is_dir() or not (dataset_root / "vali_20k").is_dir():
        raise FileNotFoundError(
            f"dataset root must contain train/ and vali_20k/: {dataset_root}"
        )
    os.environ["DREL_DATASET_ROOT"] = str(dataset_root)
    run_train(args.config, seed_override=args.seed, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
