from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecg_transformer.ptbxl import (
    SPLITS,
    TRAIN_FOLDS,
    ensure_records100,
    load_metadata,
    load_record_100hz,
)


@dataclass(frozen=True)
class LeadNorm:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, signal: np.ndarray) -> np.ndarray:
        return ((signal - self.mean[:, None]) / self.std[:, None]).astype(np.float32)


def fit_lead_norm(
    signals: np.ndarray, folds: np.ndarray | None = None
) -> LeadNorm:
    n_records = len(signals)
    if folds is None:
        index = np.arange(n_records)
    else:
        index = np.flatnonzero(np.isin(folds, list(TRAIN_FOLDS)))
    if index.size == 0:
        raise ValueError("no train signals to fit lead norm")
    n = 0
    mean = np.zeros(12, dtype=np.float64)
    m2 = np.zeros(12, dtype=np.float64)
    for i in index:
        sig = np.asarray(signals[i], dtype=np.float64)
        n_b = int(sig.shape[-1])
        mean_b = sig.mean(axis=1)
        m2_b = np.square(sig - mean_b[:, None]).sum(axis=1)
        if n == 0:
            n, mean, m2 = n_b, mean_b, m2_b
            continue
        n_ab = n + n_b
        delta = mean_b - mean
        mean = mean + delta * (n_b / n_ab)
        m2 = m2 + m2_b + np.square(delta) * n * n_b / n_ab
        n = n_ab
    std = np.maximum(np.sqrt(m2 / n), 1e-6)
    return LeadNorm(mean.astype(np.float32), std.astype(np.float32))


def apply_train_norm(
    signals: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, LeadNorm]:
    norm = fit_lead_norm(signals, folds)
    out = (
        (signals - norm.mean[None, :, None]) / norm.std[None, :, None]
    ).astype(np.float32)
    return out, norm


def _save_sidecars(
    packed_dir: Path,
    *,
    labels: np.ndarray,
    lead_mask: np.ndarray,
    ecg_id: np.ndarray,
    patient_id: np.ndarray,
    strat_fold: np.ndarray,
    lead_mean: np.ndarray,
    lead_std: np.ndarray,
) -> None:
    np.save(packed_dir / "labels.npy", np.ascontiguousarray(labels, dtype=np.float32))
    np.save(packed_dir / "lead_mask.npy", np.ascontiguousarray(lead_mask, dtype=np.float32))
    np.save(packed_dir / "ecg_id.npy", np.ascontiguousarray(ecg_id, dtype=np.int64))
    np.save(packed_dir / "patient_id.npy", np.ascontiguousarray(patient_id, dtype=np.int64))
    np.save(packed_dir / "strat_fold.npy", np.ascontiguousarray(strat_fold, dtype=np.int64))
    np.save(packed_dir / "lead_mean.npy", np.ascontiguousarray(lead_mean, dtype=np.float32))
    np.save(packed_dir / "lead_std.npy", np.ascontiguousarray(lead_std, dtype=np.float32))


def save_packed(
    packed_dir: Path,
    *,
    signals: np.ndarray,
    labels: np.ndarray,
    lead_mask: np.ndarray,
    ecg_id: np.ndarray,
    patient_id: np.ndarray,
    strat_fold: np.ndarray,
    lead_mean: np.ndarray,
    lead_std: np.ndarray,
) -> None:
    packed_dir = Path(packed_dir)
    packed_dir.mkdir(parents=True, exist_ok=True)
    np.save(packed_dir / "signals.npy", np.ascontiguousarray(signals, dtype=np.float32))
    _save_sidecars(
        packed_dir,
        labels=labels,
        lead_mask=lead_mask,
        ecg_id=ecg_id,
        patient_id=patient_id,
        strat_fold=strat_fold,
        lead_mean=lead_mean,
        lead_std=lead_std,
    )


def pack_normalized_corpus(root: Path, packed_dir: Path) -> Path:
    root = Path(root)
    packed_dir = Path(packed_dir)
    meta, labels = load_metadata(root)
    names = ensure_records100(meta["filename_lr"])
    missing = [name for name in names if not (root / f"{name}.dat").exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} records100 .dat files")
    n = len(meta)
    packed_dir.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        packed_dir / "signals.npy", mode="w+", dtype=np.float32, shape=(n, 12, 1000)
    )
    try:
        for i, rel in enumerate(names):
            mm[i] = load_record_100hz(root / rel)
            if (i + 1) % 2000 == 0 or i + 1 == n:
                print(f"read {i + 1}/{n}", flush=True)
        mm.flush()
        folds = meta["strat_fold"].to_numpy(dtype=np.int64)
        norm = fit_lead_norm(mm, folds)
        for i in range(n):
            mm[i] = norm.transform(np.asarray(mm[i]))
            if (i + 1) % 2000 == 0 or i + 1 == n:
                print(f"norm {i + 1}/{n}", flush=True)
        mm.flush()
    finally:
        del mm
    _save_sidecars(
        packed_dir,
        labels=labels.to_numpy(dtype=np.float32),
        lead_mask=np.ones((n, 12), dtype=np.float32),
        ecg_id=meta["ecg_id"].to_numpy(dtype=np.int64),
        patient_id=meta["patient_id"].to_numpy(dtype=np.int64),
        strat_fold=meta["strat_fold"].to_numpy(dtype=np.int64),
        lead_mean=norm.mean,
        lead_std=norm.std,
    )
    return packed_dir


class PTBXLDataset(Dataset):
    def __init__(self, packed_dir: Path | str, split: str) -> None:
        try:
            folds = SPLITS[split]
        except KeyError as err:
            raise ValueError(f"split must be train, val, or test, got {split!r}") from err
        packed_dir = Path(packed_dir)
        self.signals = np.load(packed_dir / "signals.npy", mmap_mode="r")
        self.labels = np.load(packed_dir / "labels.npy", mmap_mode="r")
        self.lead_mask = np.load(packed_dir / "lead_mask.npy", mmap_mode="r")
        self.ecg_ids = np.load(packed_dir / "ecg_id.npy", mmap_mode="r")
        self.patient_ids = np.load(packed_dir / "patient_id.npy", mmap_mode="r")
        strat_fold = np.load(packed_dir / "strat_fold.npy", mmap_mode="r")
        self._index = np.flatnonzero(np.isin(strat_fold, list(folds)))

    def __len__(self) -> int:
        return int(self._index.size)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        j = int(self._index[i])
        return {
            "signal": torch.from_numpy(np.array(self.signals[j], copy=True)),
            "label": torch.from_numpy(np.array(self.labels[j], copy=True)),
            "lead_mask": torch.from_numpy(np.array(self.lead_mask[j], copy=True)),
            "patient_id": int(self.patient_ids[j]),
            "ecg_id": int(self.ecg_ids[j]),
        }
