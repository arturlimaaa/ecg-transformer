import ast
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def diagnostic_statement_map(scp_statements: pd.DataFrame) -> dict[str, str]:
    diagnostic = scp_statements[scp_statements["diagnostic"] == 1]
    return diagnostic["diagnostic_class"].to_dict()


def superclass_vector(
    scp_codes: dict[str, float], stmt_map: dict[str, str]
) -> np.ndarray:
    labels = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    index = {name: i for i, name in enumerate(SUPERCLASSES)}
    for code in scp_codes:
        cls = stmt_map.get(code)
        if cls in index:
            labels[index[cls]] = 1.0
    return labels


def patient_fold_overlap_count(meta: pd.DataFrame) -> int:
    folds_per_patient = meta.groupby("patient_id")["strat_fold"].nunique()
    return int((folds_per_patient > 1).sum())


def select_debug_records(
    meta: pd.DataFrame,
    labels: pd.DataFrame,
    n: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    if len(meta) < n:
        raise ValueError(f"need at least {n} records, got {len(meta)}")
    labels = labels.loc[meta.index]
    required: list[int] = []

    sizes = meta.groupby("patient_id").size()
    multi = sizes[sizes >= 2].sort_values().index.tolist()
    if len(multi) < 2:
        raise ValueError("need two patients with multiple ECGs")
    for pid in multi[:2]:
        required.extend(meta.index[meta["patient_id"] == pid].tolist())

    cooccur = labels.index[labels.sum(axis=1) >= 2].tolist()
    if not cooccur:
        raise ValueError("need a record with co-occurring superclasses")
    if not any(i in required for i in cooccur):
        required.append(cooccur[0])

    for cls in SUPERCLASSES:
        if cls == "NORM":
            continue
        hits = labels.index[labels[cls] == 1].tolist()
        if hits and not any(i in required for i in hits):
            required.append(hits[0])

    seen: set[int] = set()
    ordered: list[int] = []
    for i in required:
        if i in seen:
            continue
        seen.add(i)
        ordered.append(i)
    if len(ordered) > n:
        raise ValueError("required records exceed n")

    remaining = [i for i in meta.index.tolist() if i not in seen]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    ordered.extend(remaining[: n - len(ordered)])
    return meta.loc[ordered]


def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(root / "ptbxl_database.csv")
    stmts = pd.read_csv(root / "scp_statements.csv", index_col=0)
    stmt_map = diagnostic_statement_map(stmts)
    codes = meta["scp_codes"].map(ast.literal_eval)
    stacked = np.stack([superclass_vector(c, stmt_map) for c in codes])
    labels = pd.DataFrame(stacked, columns=list(SUPERCLASSES), index=meta.index)
    return meta, labels


LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
S3_BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def load_record_100hz(path: Path) -> np.ndarray:
    import wfdb

    signal, fields = wfdb.rdsamp(str(path))
    if fields["fs"] != 100:
        raise ValueError(f"expected 100 Hz, got {fields['fs']}")
    if signal.shape != (1000, 12):
        raise ValueError(f"expected (1000, 12), got {signal.shape}")
    if tuple(fields["sig_name"]) != LEADS:
        raise ValueError(f"unexpected leads {fields['sig_name']}")
    return np.ascontiguousarray(signal.T, dtype=np.float32)


def download_records100(filenames_lr: Iterable[str], root: Path) -> None:
    rels: list[str] = []
    for name in filenames_lr:
        if not name.startswith("records100/") or "records500" in name:
            raise ValueError(f"only records100 paths are allowed, got {name}")
        rels.append(f"{name}.hea")
        rels.append(f"{name}.dat")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda rel: _fetch_record(rel, root), rels))


def _fetch_record(rel: str, root: Path) -> None:
    dest = root / rel
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(S3_BASE + rel, dest)
