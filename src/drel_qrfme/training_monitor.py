"""训练过程的标准化可视化与机器可读日志。

本模块不参与模型前向、损失或参数更新，只负责把已经计算出的指标写到：

* ``metrics.csv``：每个完整 epoch 一行，Excel 也能直接打开；
* ``latest_metrics.json``：最近一个完整 epoch 的简明状态；
* ``tensorboard/``：供 TensorBoard 绘制实时曲线；
* ``live_progress.json``：当前 epoch 内最近一次批次进度。

因此，即使监控代码出现问题，也不应该改变算法的数值结果。
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping


CSV_FIELDS = (
    "epoch",
    "train_loss",
    "train_top1",
    "learning_rate",
    "val_loss",
    "val_top1",
    "val_macro_f1",
    "val_weighted_f1",
    "val_bottom5_mean_f1",
    "val_min_class_f1",
    "val_mean_precision",
    "val_mean_recall",
    "val_nll",
    "val_ece_15",
    "val_brier_score",
    "checkpoint_score",
    "is_best",
)


def _finite_float(value: Any) -> float | None:
    """仅接受可写入曲线的有限数；字符串、NaN 和无穷大均忽略。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _checkpoint_score(value: Any) -> float | None:
    """兼容加权分数(float)和旧版 ``(macro_f1, top1)`` 排序键。"""

    direct = _finite_float(value)
    if direct is not None:
        return direct
    if isinstance(value, (list, tuple)) and value:
        return _finite_float(value[0])
    return None


def _row_from_history(item: Mapping[str, Any], *, is_best: bool = False) -> dict[str, Any]:
    train = item.get("train") if isinstance(item.get("train"), Mapping) else {}
    val = item.get("val") if isinstance(item.get("val"), Mapping) else {}
    return {
        "epoch": int(item.get("epoch", 0)),
        "train_loss": _finite_float(train.get("loss")),
        "train_top1": _finite_float(train.get("top1")),
        "learning_rate": _finite_float(train.get("lr")),
        "val_loss": _finite_float(val.get("loss")),
        "val_top1": _finite_float(val.get("top1")),
        "val_macro_f1": _finite_float(val.get("macro_f1")),
        "val_weighted_f1": _finite_float(val.get("weighted_f1")),
        "val_bottom5_mean_f1": _finite_float(val.get("bottom5_mean_f1")),
        "val_min_class_f1": _finite_float(val.get("min_class_f1")),
        "val_mean_precision": _finite_float(val.get("mean_precision")),
        "val_mean_recall": _finite_float(val.get("mean_recall")),
        "val_nll": _finite_float(val.get("nll")),
        "val_ece_15": _finite_float(val.get("ece_15")),
        "val_brier_score": _finite_float(val.get("brier_score")),
        "checkpoint_score": _checkpoint_score(item.get("checkpoint_score")),
        "is_best": int(bool(is_best)),
    }


class TrainingMonitor:
    """把终端之外的训练状态持续写入输出目录。

    ``history`` 可包含从 checkpoint 一起迁移来的 1--49 轮记录。首次启动时
    会把它们补写到 CSV 和 TensorBoard，所以续训者一打开页面就能看到完整曲线。
    """

    def __init__(self, output_dir: Path, history: list[dict[str, Any]]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.latest_path = self.output_dir / "latest_metrics.json"
        self.live_path = self.output_dir / "live_progress.json"
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self._rows = [_row_from_history(item) for item in history]
        self._mark_historical_best()
        self._rewrite_csv()

        # 延迟导入：普通指标文件仍可在 TensorBoard 包临时损坏时生成。
        try:
            from torch.utils.tensorboard import SummaryWriter
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "无法启用 TensorBoard。请重新执行环境安装脚本，或安装 tensorboard。"
            ) from exc
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir), flush_secs=15)

        marker = self.tensorboard_dir / ".historical_metrics_backfilled"
        if not marker.exists():
            for row in self._rows:
                self._write_epoch_scalars(row)
            self.writer.flush()
            marker.write_text("Historical epoch metrics were written once.\n", encoding="utf-8")

    def _mark_historical_best(self) -> None:
        valid = [row for row in self._rows if row["checkpoint_score"] is not None]
        if valid:
            max(valid, key=lambda row: float(row["checkpoint_score"]))["is_best"] = 1

    def _rewrite_csv(self) -> None:
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        try:
            temporary.replace(self.csv_path)
        except PermissionError:
            # Windows Excel 可能锁住已打开的 CSV。监控失败绝不能中断模型训练；
            # 关闭 Excel 后，下一个 epoch 会再次写入完整表格。
            temporary.unlink(missing_ok=True)
            warnings.warn(
                f"metrics.csv 正被其他程序占用，本轮暂未更新：{self.csv_path}",
                RuntimeWarning,
            )

    def _write_epoch_scalars(self, row: Mapping[str, Any]) -> None:
        epoch = int(row["epoch"])
        tag_by_column = {
            "train_loss": "train/loss",
            "train_top1": "train/top1",
            "learning_rate": "train/learning_rate",
            "val_loss": "validation/loss",
            "val_top1": "validation/top1",
            "val_macro_f1": "validation/macro_f1",
            "val_weighted_f1": "validation/weighted_f1",
            "val_bottom5_mean_f1": "validation/bottom5_mean_f1",
            "val_min_class_f1": "validation/min_class_f1",
            "val_mean_precision": "validation/mean_precision",
            "val_mean_recall": "validation/mean_recall",
            "val_nll": "validation/nll",
            "val_ece_15": "validation/ece_15",
            "val_brier_score": "validation/brier_score",
            "checkpoint_score": "checkpoint/selection_score",
        }
        for column, tag in tag_by_column.items():
            value = _finite_float(row.get(column))
            if value is not None:
                self.writer.add_scalar(tag, value, epoch)

    def log_train_progress(
        self,
        *,
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        top1: float,
        learning_rate: float,
    ) -> None:
        """在每个 ``log_every_steps`` 节点记录 epoch 内实时进度。"""

        payload = {
            "epoch": int(epoch),
            "step": int(step),
            "total_steps": int(total_steps),
            "progress_percent": 100.0 * int(step) / max(int(total_steps), 1),
            "running_loss": float(loss),
            "running_top1": float(top1),
            "learning_rate": float(learning_rate),
            "note": "running_* 是本轮截至当前批次的临时值，不是完整 epoch 结果。",
        }
        temporary = self.live_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.live_path)

        # 让不同 epoch 的 batch 曲线拥有单调递增的横轴。
        global_step = (int(epoch) - 1) * int(total_steps) + int(step)
        self.writer.add_scalar("train_step/running_loss", float(loss), global_step)
        self.writer.add_scalar("train_step/running_top1", float(top1), global_step)
        self.writer.add_scalar("train_step/learning_rate", float(learning_rate), global_step)
        self.writer.flush()

    def log_epoch(self, item: Mapping[str, Any], *, is_best: bool) -> None:
        """验证结束后写入一个完整 epoch；此时数值才是论文可引用指标。"""

        row = _row_from_history(item, is_best=is_best)
        self._rows = [old for old in self._rows if int(old["epoch"]) != int(row["epoch"])]
        if is_best:
            # CSV 中只标记“当前总冠军”，不保留已经被超越的旧 best 标记。
            for old in self._rows:
                old["is_best"] = 0
        self._rows.append(row)
        self._rows.sort(key=lambda value: int(value["epoch"]))
        self._rewrite_csv()
        self._write_epoch_scalars(row)
        self.writer.flush()

        payload = {
            "epoch": int(row["epoch"]),
            "is_best": bool(is_best),
            "metrics": {key: value for key, value in row.items() if key not in {"epoch", "is_best"}},
            "explanation": {
                "top1": "全部验证样本中预测正确的比例，越高越好。",
                "macro_f1": "27 个类别的 F1 等权平均，越高越好。",
                "bottom5_mean_f1": "本轮最差 5 类的平均 F1，越高越好。",
                "checkpoint_score": "用于决定 best_checkpoint 的综合分数，越高越好。",
            },
        }
        temporary = self.latest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.latest_path)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
