from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import re
from typing import Iterator, Mapping

import pandas as pd
import torch
from torch.utils.data import Sampler


_LEADING_TIMESTAMP = re.compile(r"^(\d{8})(\d{4})(\d{2})?")
_EXACT_SECOND = re.compile(r"^\d{14}$")


class RSCDTimestampError(ValueError):
    """Raised when an RSCD acquisition second cannot be recovered safely."""


def extract_rscd_acquisition_second(value: object) -> tuple[str, str, str]:
    """Return ``YYYYMMDDHHMMSS``, date, and source timestamp precision.

    Normal RSCD filenames begin with at least 14 timestamp digits.  The six
    known minute-precision training names are represented at second ``00``;
    this matches the leakage-safe hourly-protocol builder.
    """

    text = str(value or "").replace("\\", "/")
    basename = text.rsplit("/", maxsplit=1)[-1]
    match = _LEADING_TIMESTAMP.match(basename)
    if match is None:
        raise RSCDTimestampError(
            f"RSCD path has no leading YYYYMMDDHHMM timestamp: {value}"
        )
    date, hour_minute, seconds = match.groups()
    precision = "second" if seconds is not None else "minute"
    acquisition_second = f"{date}{hour_minute}{seconds or '00'}"
    try:
        datetime.strptime(acquisition_second, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise RSCDTimestampError(
            f"invalid RSCD leading timestamp {acquisition_second!r}: {value}"
        ) from exc
    return acquisition_second, date, precision


def _validate_manifest_second(value: object) -> str:
    text = str(value or "").strip()
    if not _EXACT_SECOND.fullmatch(text):
        raise RSCDTimestampError(
            "sampling_acquisition_second must be exactly YYYYMMDDHHMMSS, "
            f"got {value!r}"
        )
    try:
        datetime.strptime(text, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise RSCDTimestampError(
            f"invalid sampling_acquisition_second {text!r}"
        ) from exc
    return text


def _stable_u64(value: str, seed: int) -> int:
    payload = f"{int(seed)}|{value}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


class EpochClassSecondRotationSampler(Sampler[int]):
    """Select at most one frame per RSCD class/acquisition-second each epoch.

    A seed-specific stable order is built inside every same-class burst.  Epoch
    ``e`` selects offset ``(e - 1) mod burst_size``, so repeated epochs rotate
    through all frames instead of permanently discarding near-duplicate video
    frames.  The resulting representatives are shuffled deterministically.

    The sampler owns only indices: the dataset keeps the complete manifest.
    ``start_index`` slices the deterministic *local-rank* sequence and therefore
    supports exact batch-boundary checkpoint resume without serializing a
    mutable iterator cursor.
    """

    _VALID_MISSING_POLICIES = frozenset({"error", "unique"})

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        seed: int,
        epoch: int,
        start_index: int = 0,
        missing_timestamp: str = "error",
        max_samples_per_class: int | None = None,
        samples_per_class: Mapping[str, int] | None = None,
        max_samples: int | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if "image_path" not in frame or "class_label_canonical" not in frame:
            raise ValueError(
                "rotation sampler requires image_path and "
                "class_label_canonical columns"
            )
        duplicated_paths = frame["image_path"].astype(str).duplicated(keep=False)
        if bool(duplicated_paths.any()):
            examples = (
                frame.loc[duplicated_paths, "image_path"].astype(str).head(5).tolist()
            )
            raise ValueError(
                "rotation sampler requires unique image paths; "
                f"duplicate examples={examples}"
            )
        policy = str(missing_timestamp).strip().lower()
        if policy not in self._VALID_MISSING_POLICIES:
            raise ValueError(
                "missing_timestamp must be one of "
                f"{sorted(self._VALID_MISSING_POLICIES)}, got {policy!r}"
            )
        if max_samples_per_class is not None and int(max_samples_per_class) <= 0:
            raise ValueError("max_samples_per_class must be positive or None")
        if max_samples_per_class is not None and samples_per_class is not None:
            raise ValueError(
                "max_samples_per_class and samples_per_class are mutually exclusive"
            )
        normalized_samples_per_class = (
            None
            if samples_per_class is None
            else {str(label): int(value) for label, value in samples_per_class.items()}
        )
        if normalized_samples_per_class is not None and any(
            value <= 0 for value in normalized_samples_per_class.values()
        ):
            raise ValueError("every samples_per_class value must be positive")
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError("max_samples must be positive or None")

        inferred_replicas, inferred_rank = self._distributed_context()
        self.num_replicas = int(
            inferred_replicas if num_replicas is None else num_replicas
        )
        self.rank = int(inferred_rank if rank is None else rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )

        self.seed = int(seed)
        self.epoch = int(epoch)
        self.missing_timestamp = policy
        self.max_samples_per_class = (
            None if max_samples_per_class is None else int(max_samples_per_class)
        )
        self.samples_per_class = normalized_samples_per_class
        self.max_samples = None if max_samples is None else int(max_samples)
        self._groups = self._build_groups(frame)
        if self.samples_per_class is not None:
            observed = set(self._groups)
            supplied = set(self.samples_per_class)
            if observed != supplied:
                raise ValueError(
                    "samples_per_class must contain every observed class exactly once; "
                    f"missing={sorted(observed-supplied)} "
                    f"unknown={sorted(supplied-observed)}"
                )
            impossible = {
                label: (self.samples_per_class[label], len(groups))
                for label, groups in self._groups.items()
                if self.samples_per_class[label] > len(groups)
            }
            if impossible:
                raise ValueError(
                    "samples_per_class requests more class-second groups than exist: "
                    f"{impossible}"
                )
        self.start_index = min(
            max(int(start_index), 0),
            self._local_size_for_epoch(self.epoch),
        )

    @staticmethod
    def _distributed_context() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _build_groups(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, tuple[tuple[str, tuple[int, ...]], ...]]:
        grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
        has_manifest_second = "sampling_acquisition_second" in frame.columns
        missing_examples: list[str] = []
        labels = frame["class_label_canonical"].astype(str).tolist()
        image_paths = frame["image_path"].astype(str).tolist()
        supplied_seconds = (
            frame["sampling_acquisition_second"].tolist()
            if has_manifest_second
            else [None] * len(frame)
        )

        for row_index, (label, image_path, supplied_second) in enumerate(
            zip(labels, image_paths, supplied_seconds, strict=True)
        ):
            second: str | None = None
            if supplied_second is not None and not pd.isna(supplied_second) and str(
                supplied_second
            ).strip():
                try:
                    second = _validate_manifest_second(supplied_second)
                except RSCDTimestampError:
                    if self.missing_timestamp == "error":
                        raise
            if second is None:
                try:
                    second = extract_rscd_acquisition_second(image_path)[0]
                except RSCDTimestampError:
                    if self.missing_timestamp == "error":
                        missing_examples.append(image_path)
                        continue
                    # A missing timestamp is deliberately made a singleton.  It
                    # may be sampled, but can never be merged with another row.
                    second = f"__missing__{row_index:012d}"
            grouped[(label, second)].append((int(row_index), image_path))

        if missing_examples:
            preview = missing_examples[:5]
            raise RSCDTimestampError(
                "rotation sampler could not recover acquisition seconds for "
                f"{len(missing_examples)} row(s); examples={preview}. Set "
                "train.temporal_rotation_missing_timestamp=unique only when "
                "singleton fallback is intentional."
            )
        if not grouped:
            raise ValueError("rotation sampler found no rows")

        by_class: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
        for (label, second), rows in grouped.items():
            rows.sort(
                key=lambda item: (
                    _stable_u64(f"frame|{item[1]}", self.seed),
                    item[1],
                    item[0],
                )
            )
            by_class[label].append(
                (second, tuple(row_index for row_index, _ in rows))
            )
        result: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {}
        for label, groups in by_class.items():
            groups.sort(
                key=lambda item: (
                    _stable_u64(f"group|{label}|{item[0]}", self.seed),
                    item[0],
                )
            )
            result[label] = tuple(groups)
        return result

    @staticmethod
    def _epoch_offset(epoch: int) -> int:
        # Training epochs are one-based.  Accept epoch zero for unit tests and
        # evaluation utilities without producing a negative rotation.
        return max(int(epoch) - 1, 0)

    def _selected_global_indices(self, epoch: int) -> list[int]:
        epoch_offset = self._epoch_offset(epoch)
        selected: list[int] = []
        for label in sorted(self._groups):
            groups = self._groups[label]
            # ``samples_per_class`` preserves a frozen control distribution
            # exactly.  This is important when the experiment intends to
            # isolate temporal diversity rather than silently rebalance labels.
            limit = (
                self.samples_per_class[label]
                if self.samples_per_class is not None
                else self.max_samples_per_class
            )
            if limit is not None and len(groups) > limit:
                group_start = (epoch_offset * limit) % len(groups)
                chosen_groups = [
                    groups[(group_start + offset) % len(groups)]
                    for offset in range(limit)
                ]
            else:
                chosen_groups = list(groups)
            for _second, row_indices in chosen_groups:
                selected.append(row_indices[epoch_offset % len(row_indices)])

        generator = torch.Generator(device="cpu")
        # Keep the seed inside torch's signed 64-bit range while mixing seed and
        # epoch independently of Python's randomized hash implementation.
        order_seed = _stable_u64(f"epoch-order|{int(epoch)}", self.seed) % (2**63 - 1)
        generator.manual_seed(order_seed)
        permutation = torch.randperm(len(selected), generator=generator).tolist()
        selected = [selected[index] for index in permutation]
        if self.max_samples is not None:
            selected = selected[: self.max_samples]
        return selected

    def _local_indices(self, epoch: int) -> list[int]:
        selected = self._selected_global_indices(epoch)
        if self.num_replicas > 1:
            # Never pad by repeating samples: truncate at most world_size - 1
            # representatives so the global one-per-key invariant is retained
            # and every rank executes the same number of batches.
            divisible_size = (len(selected) // self.num_replicas) * self.num_replicas
            selected = selected[:divisible_size]
            selected = selected[self.rank : divisible_size : self.num_replicas]
        return selected

    def _local_size_for_epoch(self, epoch: int) -> int:
        global_size = sum(
            min(
                len(groups),
                (
                    self.samples_per_class[label]
                    if self.samples_per_class is not None
                    else self.max_samples_per_class
                )
                if (
                    self.samples_per_class is not None
                    or self.max_samples_per_class is not None
                )
                else len(groups),
            )
            for label, groups in self._groups.items()
        )
        if self.max_samples is not None:
            global_size = min(global_size, self.max_samples)
        if self.num_replicas > 1:
            global_size = global_size // self.num_replicas
        return int(global_size)

    @property
    def global_samples_per_epoch(self) -> int:
        """Number selected before distributed truncation/sharding."""

        return len(self._selected_global_indices(self.epoch))

    @property
    def class_second_groups(self) -> int:
        return sum(len(groups) for groups in self._groups.values())

    def indices_for_epoch(self, epoch: int | None = None) -> list[int]:
        """Return the unsliced local-rank sequence for audits and tests."""

        return self._local_indices(self.epoch if epoch is None else int(epoch))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.start_index = 0

    def __iter__(self) -> Iterator[int]:
        return iter(self._local_indices(self.epoch)[self.start_index :])

    def __len__(self) -> int:
        return self._local_size_for_epoch(self.epoch) - self.start_index
