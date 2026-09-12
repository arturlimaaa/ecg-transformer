import ast
import time
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
TRAIN_FOLDS = frozenset(range(1, 9))
VAL_FOLDS = frozenset({9})
TEST_FOLDS = frozenset({10})
SPLITS = {"train": TRAIN_FOLDS, "val": VAL_FOLDS, "test": TEST_FOLDS}


def official_split(meta: pd.DataFrame, split: str) -> pd.DataFrame:
    try:
        folds = SPLITS[split]
    except KeyError as err:
        raise ValueError(f"split must be train, val, or test, got {split!r}") from err
    return meta[meta["strat_fold"].isin(folds)].copy()


def patient_id_set(meta: pd.DataFrame) -> set[int]:
    return {int(pid) for pid in meta["patient_id"]}


def diagnostic_statement_map(scp_statements: pd.DataFrame) -> dict[str, str]:
    diagnostic = scp_statements[scp_statements["diagnostic"] == 1]
    return diagnostic["diagnostic_class"].to_dict()


def superclass_vector(
    scp_codes: dict[str, float], stmt_map: dict[str, str]
) -> np.ndarray:
    """Aggregate diagnostic SCP codes onto the five PTB-XL superclasses.

    Any diagnostic code present in ``scp_codes`` counts, including likelihood
    0. That matches PhysioNet's published superclass table; do not drop
    zero-likelihood statements.
    """
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


def ensure_records100(filenames_lr: Iterable[str]) -> list[str]:
    names: list[str] = []
    for name in filenames_lr:
        if not name.startswith("records100/") or "records500" in name:
            raise ValueError(f"only records100 paths are allowed, got {name}")
        names.append(name)
    return names


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


def download_records100(
    filenames_lr: Iterable[str], root: Path, max_workers: int = 8
) -> None:
    rels: list[str] = []
    for name in ensure_records100(filenames_lr):
        rels.append(f"{name}.hea")
        rels.append(f"{name}.dat")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda rel: _fetch_record(rel, root), rels))


def _fetch_record(rel: str, root: Path, retries: int = 5) -> None:
    dest = root / rel
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = S3_BASE + rel
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as src, open(tmp, "wb") as out:
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(dest)
            return
        except Exception as err:
            last_err = err
            if tmp.exists():
                tmp.unlink()
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {rel}: {last_err}") from last_err
