from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_transformer.dataset import PTBXLDataset, apply_train_norm, fit_lead_norm, save_packed
from ecg_transformer.ptbxl import (
    load_metadata,
    load_record_100hz,
    official_split,
    patient_id_set,
    select_debug_records,
)

PTBXL = Path("data/ptb-xl")


def test_official_split_patient_sets_are_disjoint_by_set_intersection():
    meta = pd.DataFrame(
        {
            "ecg_id": [10, 11, 12, 13, 14],
            "patient_id": [1, 1, 2, 3, 4],
            "strat_fold": [1, 8, 9, 10, 2],
        }
    )
    train = official_split(meta, "train")
    val = official_split(meta, "val")
    test = official_split(meta, "test")
    train_p = patient_id_set(train)
    val_p = patient_id_set(val)
    test_p = patient_id_set(test)
    assert train_p & val_p == set()
    assert train_p & test_p == set()
    assert val_p & test_p == set()
    assert train_p == {1, 4}
    assert val_p == {2}
    assert test_p == {3}
    assert set(train["strat_fold"].tolist()) <= set(range(1, 9))
    assert val["strat_fold"].tolist() == [9]
    assert test["strat_fold"].tolist() == [10]


@pytest.mark.skipif(
    not (PTBXL / "ptbxl_database.csv").exists(), reason="PTB-XL metadata missing"
)
def test_real_official_splits_cover_all_records_with_disjoint_patients():
    meta, _ = load_metadata(PTBXL)
    train = official_split(meta, "train")
    val = official_split(meta, "val")
    test = official_split(meta, "test")
    train_p = patient_id_set(train)
    val_p = patient_id_set(val)
    test_p = patient_id_set(test)
    assert train_p & val_p == set()
    assert train_p & test_p == set()
    assert val_p & test_p == set()
    assert len(train) + len(val) + len(test) == 21799
    assert len(meta) == 21799


@pytest.mark.skipif(
    not (PTBXL / "ptbxl_database.csv").exists(), reason="PTB-XL metadata missing"
)
def test_unlabeled_diagnostic_rows_are_kept_as_all_zero():
    meta, labels = load_metadata(PTBXL)
    empty = labels.to_numpy().sum(axis=1) == 0
    assert int(empty.sum()) == 411
    assert len(meta) == len(labels) == 21799


def test_held_out_fold_does_not_change_train_lead_mean():
    train = np.ones((4, 12, 1000), dtype=np.float32)
    val = np.full((2, 12, 1000), 50.0, dtype=np.float32)
    signals = np.concatenate([train, val], axis=0)
    folds = np.array([1, 2, 7, 8, 9, 9], dtype=np.int64)
    train_norm = fit_lead_norm(signals, folds)
    leaked = fit_lead_norm(signals)
    assert np.allclose(train_norm.mean, 1.0)
    assert not np.allclose(leaked.mean, train_norm.mean)


def test_val_records_use_train_lead_stats_not_their_own():
    rng = np.random.default_rng(0)
    train = rng.normal(3.0, 2.0, size=(8, 12, 1000)).astype(np.float32)
    val = rng.normal(9.0, 2.0, size=(4, 12, 1000)).astype(np.float32)
    signals = np.concatenate([train, val], axis=0)
    folds = np.array([1, 1, 2, 3, 4, 5, 6, 8, 9, 9, 9, 9], dtype=np.int64)
    out, norm = apply_train_norm(signals, folds)
    train_out = out[:8]
    val_out = out[8:]
    assert np.allclose(train_out.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(train_out.std(axis=(0, 2)), 1.0, atol=1e-4)
    assert np.mean(np.abs(val_out.mean(axis=(0, 2)))) > 1.0
    own = fit_lead_norm(val)
    assert not np.allclose(own.mean, norm.mean)


def _tiny_corpus() -> dict[str, np.ndarray]:
    n = 6
    signals = np.zeros((n, 12, 1000), dtype=np.float32)
    signals[:4] = 1.0
    signals[4] = 3.0
    signals[5] = 5.0
    labels = np.zeros((n, 5), dtype=np.float32)
    labels[0] = [1, 0, 0, 0, 0]
    labels[1] = [0, 1, 0, 0, 0]
    labels[2] = [0, 0, 0, 0, 0]
    labels[3] = [0, 0, 1, 0, 1]
    labels[4] = [1, 0, 0, 0, 0]
    labels[5] = [0, 1, 0, 0, 0]
    return {
        "signals": signals,
        "labels": labels,
        "lead_mask": np.ones((n, 12), dtype=np.float32),
        "ecg_id": np.array([101, 102, 103, 104, 105, 106], dtype=np.int64),
        "patient_id": np.array([11, 12, 13, 14, 15, 16], dtype=np.int64),
        "strat_fold": np.array([1, 2, 8, 3, 9, 10], dtype=np.int64),
    }


def test_dataset_item_has_expected_keys_and_shapes(tmp_path: Path):
    corpus = _tiny_corpus()
    save_packed(tmp_path, **corpus, lead_mean=np.zeros(12), lead_std=np.ones(12))
    item = PTBXLDataset(tmp_path, "train")[0]
    assert set(item) == {"signal", "label", "lead_mask", "patient_id", "ecg_id"}
    assert item["signal"].shape == (12, 1000)
    assert item["label"].shape == (5,)
    assert item["lead_mask"].shape == (12,)
    assert item["signal"].dtype == torch.float32
    assert item["label"].dtype == torch.float32
    assert item["lead_mask"].dtype == torch.float32
    assert torch.equal(item["lead_mask"], torch.ones(12))
    assert isinstance(item["patient_id"], int)
    assert isinstance(item["ecg_id"], int)


def test_dataset_keeps_all_zero_label_rows(tmp_path: Path):
    corpus = _tiny_corpus()
    save_packed(tmp_path, **corpus, lead_mean=np.zeros(12), lead_std=np.ones(12))
    ds = PTBXLDataset(tmp_path, "train")
    assert len(ds) == 4
    labels = torch.stack([ds[i]["label"] for i in range(len(ds))])
    assert (labels.sum(dim=1) == 0).any()


def _records100_complete() -> bool:
    if not (PTBXL / "ptbxl_database.csv").exists():
        return False
    meta, _ = load_metadata(PTBXL)
    return not any(
        not (PTBXL / f"{name}.dat").exists() for name in meta["filename_lr"]
    )


@pytest.mark.skipif(not (PTBXL / "ptbxl_database.csv").exists(), reason="PTB-XL metadata missing")
def test_random_non_debug_records_are_twelve_by_thousand():
    if not _records100_complete():
        pytest.skip("full records100 tree missing")
    meta, labels = load_metadata(PTBXL)
    debug = set(select_debug_records(meta, labels, n=100, seed=0).index)
    others = meta.index.difference(debug).to_numpy()
    sample = np.random.default_rng(1).choice(others, size=8, replace=False)
    for idx in sample:
        signal = load_record_100hz(PTBXL / meta.loc[idx, "filename_lr"])
        assert signal.shape == (12, 1000)
        assert signal.dtype == np.float32
    assert len(meta) == 21799


PACKED = PTBXL / "packed"


@pytest.mark.skipif(
    not (PACKED / "signals.npy").exists(), reason="packed corpus missing"
)
def test_packed_dataset_covers_official_folds_and_keeps_unlabeled_rows():
    meta, labels = load_metadata(PTBXL)
    train = PTBXLDataset(PACKED, "train")
    val = PTBXLDataset(PACKED, "val")
    test = PTBXLDataset(PACKED, "test")
    assert len(train) == len(official_split(meta, "train"))
    assert len(val) == len(official_split(meta, "val"))
    assert len(test) == len(official_split(meta, "test"))
    assert len(train) + len(val) + len(test) == 21799
    packed_labels = np.load(PACKED / "labels.npy", mmap_mode="r")
    assert int((packed_labels.sum(axis=1) == 0).sum()) == 411
    assert int((labels.to_numpy().sum(axis=1) == 0).sum()) == 411
    item = test[0]
    assert item["signal"].shape == (12, 1000)
    assert item["label"].shape == (5,)
    assert torch.equal(item["lead_mask"], torch.ones(12))
    mean = np.load(PACKED / "lead_mean.npy")
    std = np.load(PACKED / "lead_std.npy")
    assert mean.shape == (12,)
    assert std.shape == (12,)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)
