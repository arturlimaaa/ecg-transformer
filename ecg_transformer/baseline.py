from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_transformer.dataset import PTBXLDataset
from ecg_transformer.ptbxl import SUPERCLASSES


def train_prevalence(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2:
        raise ValueError(f"labels must be (n, classes), got {labels.shape}")
    return labels.mean(axis=0)


def constant_scores(n: int, rates: np.ndarray) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float32)
    return np.broadcast_to(rates, (n, rates.shape[-1])).copy()


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | list[float]]:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim == 1:
        labels = labels[:, None]
        scores = scores[:, None]
    n_classes = labels.shape[1]
    auroc = [
        float(roc_auc_score(labels[:, k], scores[:, k])) for k in range(n_classes)
    ]
    auprc = [
        float(average_precision_score(labels[:, k], scores[:, k]))
        for k in range(n_classes)
    ]
    return {
        "per_class_auroc": auroc,
        "macro_auroc": float(np.mean(auroc)),
        "per_class_auprc": auprc,
        "macro_auprc": float(np.mean(auprc)),
    }


def _split_labels(ds: PTBXLDataset) -> np.ndarray:
    return np.asarray(ds.labels[ds._index], dtype=np.float64)


def majority_metrics(packed_dir: Path | str) -> dict:
    train = PTBXLDataset(packed_dir, "train")
    test = PTBXLDataset(packed_dir, "test")
    rates = train_prevalence(_split_labels(train))
    scores = constant_scores(len(test), rates)
    rank = ranking_metrics(_split_labels(test), scores)
    names = SUPERCLASSES
    return {
        "n_test": len(test),
        "per_class_train_prevalence": {
            name: float(rate) for name, rate in zip(names, rates, strict=True)
        },
        "per_class_auroc": {
            name: auc for name, auc in zip(names, rank["per_class_auroc"], strict=True)
        },
        "macro_auroc": rank["macro_auroc"],
        "per_class_auprc": {
            name: ap for name, ap in zip(names, rank["per_class_auprc"], strict=True)
        },
        "macro_auprc": rank["macro_auprc"],
        "hard_majority": {
            name: int(rate >= 0.5) for name, rate in zip(names, rates, strict=True)
        },
    }
