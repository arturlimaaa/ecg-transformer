import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_transformer.baseline import constant_scores, ranking_metrics, train_prevalence
from ecg_transformer.ptbxl import SUPERCLASSES, TRAIN_FOLDS

METRICS_JSON = Path("figures/m4_metrics.json")
PACKED = Path("data/ptb-xl/packed")


def test_train_prevalence_matches_empirical_class_means():
    labels = np.array(
        [
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    rates = train_prevalence(labels)
    assert rates.shape == (5,)
    assert np.allclose(rates, labels.mean(axis=0))


def test_constant_score_vector_has_auroc_half_on_balanced_toy():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    scores = np.full(len(y), 0.37, dtype=np.float32)
    metrics = ranking_metrics(y[:, None], scores[:, None])
    assert metrics["macro_auroc"] == pytest.approx(0.5)


def test_constant_scores_are_produced_without_test_labels():
    train = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    test = np.zeros((6, 5), dtype=np.float32)
    test[:, 4] = 1.0
    rates = train_prevalence(train)
    scores = constant_scores(len(test), rates)
    assert scores.shape == (6, 5)
    assert np.allclose(scores[0], train.mean(axis=0))
    assert not np.allclose(scores[0], test.mean(axis=0))
    params = inspect.signature(constant_scores).parameters
    assert not any(
        "label" in name.lower() or name in {"y", "y_true", "y_test"} for name in params
    )
    src = inspect.getsource(constant_scores)
    assert "label" not in src
    assert "y_test" not in src
    assert "y_true" not in src


def test_fold10_majority_json_macro_auroc_is_near_chance():
    assert METRICS_JSON.exists()
    payload = json.loads(METRICS_JSON.read_text())
    required = {
        "per_class_train_prevalence",
        "per_class_auroc",
        "macro_auroc",
        "per_class_auprc",
        "macro_auprc",
        "n_test",
    }
    assert required <= set(payload)
    for key in (
        "per_class_train_prevalence",
        "per_class_auroc",
        "per_class_auprc",
    ):
        assert list(payload[key]) == list(SUPERCLASSES)
    assert payload["n_test"] == 2198
    assert 0.45 <= payload["macro_auroc"] <= 0.55
    assert 0.45 <= payload["per_class_auroc"]["HYP"] <= 0.55
    if (PACKED / "labels.npy").exists():
        labels = np.load(PACKED / "labels.npy", mmap_mode="r")
        folds = np.load(PACKED / "strat_fold.npy", mmap_mode="r")
        train_rate = np.asarray(
            labels[np.isin(folds, list(TRAIN_FOLDS))], dtype=np.float64
        ).mean(axis=0)
        test_rate = np.asarray(labels[folds == 10], dtype=np.float64).mean(axis=0)
        for i, name in enumerate(SUPERCLASSES):
            assert payload["per_class_train_prevalence"][name] == pytest.approx(
                float(train_rate[i]), rel=1e-6, abs=1e-6
            )
        assert not np.allclose(train_rate, test_rate)
