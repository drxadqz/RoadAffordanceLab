"""在新旧电脑上评估 DREL-QRFME checkpoint；默认绝不读取 test。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from drel_qrfme.training_engine import run_eval  # noqa: E402
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DREL-QRFME on RSCD validation.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="包含 train/ 和 vali_20k/ 的 RSCD 根目录")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/drel_qrfme/rscd_full_train_seed097.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/evaluation")
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not (dataset_root / "train").is_dir() or not (dataset_root / "vali_20k").is_dir():
        raise FileNotFoundError(
            f"dataset root must contain train/ and vali_20k/: {dataset_root}"
        )
    os.environ["DREL_DATASET_ROOT"] = str(dataset_root)
    run_eval(args.config, args.checkpoint, split="val", output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
