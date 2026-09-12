from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_transformer.ptbxl import (
    SUPERCLASSES,
    diagnostic_statement_map,
    download_records100,
    load_metadata,
    load_record_100hz,
    patient_fold_overlap_count,
    select_debug_records,
    superclass_vector,
)

PTBXL = Path("data/ptb-xl")


def _stmt_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diagnostic": [1.0, 1.0, 1.0, None, None],
            "diagnostic_class": ["NORM", "MI", "HYP", None, None],
        },
        index=["NORM", "IMI", "LVH", "SR", "ABQRS"],
    )


def test_form_and_rhythm_statements_are_not_superclasses():
    stmt_map = diagnostic_statement_map(_stmt_table())
    vector = superclass_vector({"NORM": 100.0, "SR": 0.0, "ABQRS": 100.0}, stmt_map)
    assert list(vector) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert SUPERCLASSES == ("NORM", "MI", "STTC", "CD", "HYP")


def test_cooccurring_diagnostic_statements_set_multiple_superclasses():
    stmt_map = diagnostic_statement_map(_stmt_table())
    vector = superclass_vector({"IMI": 100.0, "LVH": 100.0, "SR": 0.0}, stmt_map)
    assert list(vector) == [0.0, 1.0, 0.0, 0.0, 1.0]


def test_patient_confined_to_one_fold_is_zero_overlap():
    meta = pd.DataFrame(
        {
            "patient_id": [10, 10, 11, 12],
            "strat_fold": [1, 1, 9, 10],
        }
    )
    assert patient_fold_overlap_count(meta) == 0


def test_same_patient_in_two_folds_counts_as_overlap():
    meta = pd.DataFrame(
        {
            "patient_id": [10, 10],
            "strat_fold": [1, 10],
        }
    )
    assert patient_fold_overlap_count(meta) == 1


def _skewed_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = []
    for i in range(400):
        rows.append({"ecg_id": i, "patient_id": 1000 + i, "strat_fold": 1})
        labels.append([1.0, 0.0, 0.0, 0.0, 0.0])
    rows.append({"ecg_id": 400, "patient_id": 1, "strat_fold": 2})
    rows.append({"ecg_id": 401, "patient_id": 1, "strat_fold": 2})
    labels.append([0.0, 1.0, 0.0, 0.0, 0.0])
    labels.append([0.0, 1.0, 0.0, 0.0, 0.0])
    rows.append({"ecg_id": 402, "patient_id": 2, "strat_fold": 3})
    rows.append({"ecg_id": 403, "patient_id": 2, "strat_fold": 3})
    labels.append([0.0, 1.0, 0.0, 0.0, 1.0])
    labels.append([0.0, 1.0, 0.0, 0.0, 1.0])
    rows.append({"ecg_id": 404, "patient_id": 3, "strat_fold": 4})
    labels.append([0.0, 0.0, 1.0, 0.0, 0.0])
    rows.append({"ecg_id": 405, "patient_id": 4, "strat_fold": 5})
    labels.append([0.0, 0.0, 0.0, 1.0, 0.0])
    meta = pd.DataFrame(rows)
    y = pd.DataFrame(labels, columns=list(SUPERCLASSES))
    return meta, y


def test_debug_set_is_not_all_norm_and_keeps_cooccurrence_and_repeat_patients():
    meta, labels = _skewed_catalog()
    chosen = select_debug_records(meta, labels, n=100, seed=0)
    assert len(chosen) == 100
    chosen_y = labels.loc[chosen.index]
    assert not (chosen_y["NORM"] == 1).all()
    assert (chosen_y.sum(axis=1) >= 2).any()
    assert int((chosen.groupby("patient_id").size() >= 2).sum()) >= 2


@pytest.mark.skipif(
    not (PTBXL / "ptbxl_database.csv").exists(), reason="PTB-XL metadata missing"
)
def test_official_ptbxl_folds_have_zero_patient_overlap():
    meta = pd.read_csv(PTBXL / "ptbxl_database.csv")
    assert patient_fold_overlap_count(meta) == 0


@pytest.mark.skipif(
    not (PTBXL / "scp_statements.csv").exists(), reason="PTB-XL metadata missing"
)
def test_real_scp_statements_map_diagnostic_codes_only():
    stmts = pd.read_csv(PTBXL / "scp_statements.csv", index_col=0)
    stmt_map = diagnostic_statement_map(stmts)
    assert stmt_map["IMI"] == "MI"
    assert stmt_map["NORM"] == "NORM"
    assert "SR" not in stmt_map
    assert "ABQRS" not in stmt_map
    vector = superclass_vector({"IMI": 100.0, "SR": 0.0, "ABQRS": 100.0}, stmt_map)
    assert np.array_equal(vector, [0.0, 1.0, 0.0, 0.0, 0.0])


@pytest.mark.skipif(
    not (PTBXL / "ptbxl_database.csv").exists(), reason="PTB-XL metadata missing"
)
def test_real_superclass_counts_match_physionet_table():
    meta, labels = load_metadata(PTBXL)
    assert len(meta) == 21799
    counts = {k: int(v) for k, v in labels.sum().items()}
    assert counts == {"NORM": 9514, "MI": 5469, "STTC": 5235, "CD": 4898, "HYP": 2649}


@pytest.mark.skipif(
    not (PTBXL / "ptbxl_database.csv").exists(), reason="PTB-XL metadata missing"
)
def test_real_debug_set_is_not_all_norm():
    meta, labels = load_metadata(PTBXL)
    chosen = select_debug_records(meta, labels, n=100, seed=0)
    chosen_y = labels.loc[chosen.index]
    assert len(chosen) == 100
    assert not (chosen_y["NORM"] == 1).all()
    assert (chosen_y.sum(axis=1) >= 2).any()
    assert int((chosen.groupby("patient_id").size() >= 2).sum()) >= 2


def test_download_rejects_records500():
    with pytest.raises(ValueError, match="records100"):
        download_records100(["records500/00000/00001_hr"], PTBXL)


@pytest.mark.skipif(
    not (PTBXL / "records100/00000/00001_lr.dat").exists(),
    reason="sample 100 Hz record missing",
)
def test_100hz_record_is_twelve_leads_by_one_thousand_samples():
    signal = load_record_100hz(PTBXL / "records100/00000/00001_lr")
    assert signal.shape == (12, 1000)
    assert signal.dtype == np.float32


def test_debug_waveforms_are_twelve_by_thousand():
    meta, labels = load_metadata(PTBXL)
    chosen = select_debug_records(meta, labels, n=100, seed=0)
    missing = [
        rel for rel in chosen["filename_lr"] if not (PTBXL / f"{rel}.dat").exists()
    ]
    if missing:
        pytest.skip(f"{len(missing)} debug records not downloaded")
    for rel in chosen["filename_lr"]:
        signal = load_record_100hz(PTBXL / rel)
        assert signal.shape == (12, 1000)
