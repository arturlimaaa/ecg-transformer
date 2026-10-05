import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ecg_transformer.cnn import CNN, apply_condition, with_model_block
from ecg_transformer.ptbxl import SUPERCLASSES

METRICS_JSON = Path("figures/m5_metrics.json")
TABLE_JSON = Path("figures/missing_lead_table.json")
DEMO_JSON = Path("figures/m8_demo.json")
PACKED_SIGNALS = Path("data/ptb-xl/packed/signals.npy")
PACKED_MASK = Path("data/ptb-xl/packed/lead_mask.npy")


def test_cnn_maps_batch_to_five_logits():
    logits = CNN()(torch.zeros(4, 12, 1000))
    assert logits.shape == (4, 5)


def test_cnn_loss_on_one_batch_is_finite_and_falls():
    torch.manual_seed(0)
    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.randn(8, 12, 1000)
    y = torch.randint(0, 2, (8, 5)).float()
    losses: list[float] = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert len(losses) == 20
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]


def test_fold10_cnn_json_beats_chance_and_keeps_the_val_checkpoint():
    assert METRICS_JSON.exists()
    payload = json.loads(METRICS_JSON.read_text())
    assert payload["condition"] == "clean"
    assert payload["n_test"] == 2198
    assert list(payload["per_class_auroc"]) == list(SUPERCLASSES)
    assert list(payload["per_class_auprc"]) == list(SUPERCLASSES)
    assert payload["macro_auroc"] > 0.5
    assert math.isfinite(payload["macro_auroc"])
    assert math.isfinite(payload["macro_auprc"])
    assert all(math.isfinite(v) for v in payload["per_class_auroc"].values())
    assert all(math.isfinite(v) for v in payload["per_class_auprc"].values())
    for key in (
        "seed",
        "selected_epoch",
        "optimizer",
        "lr",
        "batch_size",
        "max_epochs",
        "patience",
        "checkpoint",
    ):
        assert key in payload
    history = payload["history"]
    chosen = next(row for row in history if row["epoch"] == payload["selected_epoch"])
    assert chosen["val_macro_auroc"] == max(row["val_macro_auroc"] for row in history)
    assert Path(payload["checkpoint"]).exists()


def test_conditions_zero_named_leads_and_leave_the_input_and_packed_files():
    before = (PACKED_SIGNALS.stat().st_mtime_ns, PACKED_MASK.stat().st_mtime_ns)
    x = torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4) + 1
    original = x.clone()
    prec = apply_condition(x, "precordial_v1_v3")
    only = apply_condition(x, "i_ii_only")
    clean = apply_condition(x, "clean")
    assert torch.equal(x, original)
    assert torch.equal(prec[:, 6:9], torch.zeros(2, 3, 4))
    assert torch.equal(prec[:, :6], original[:, :6])
    assert torch.equal(prec[:, 9:], original[:, 9:])
    assert torch.equal(only[:, :2], original[:, :2])
    assert torch.equal(only[:, 2:], torch.zeros(2, 10, 4))
    assert torch.equal(clean, original)
    assert (PACKED_SIGNALS.stat().st_mtime_ns, PACKED_MASK.stat().st_mtime_ns) == before


def _model_block(table: dict, name: str) -> dict:
    return next(block for block in table["models"] if block["model"] == name)


def test_rescoring_one_model_replaces_its_block_and_keeps_the_others():
    table = {"models": [{"model": "cnn", "rows": ["old"]}, {"model": "transformer", "rows": ["tf"]}]}
    updated = with_model_block(table, {"model": "cnn", "rows": ["new"]})
    assert updated == {
        "models": [{"model": "cnn", "rows": ["new"]}, {"model": "transformer", "rows": ["tf"]}]
    }


def test_scoring_a_new_model_appends_its_block():
    table = {"models": [{"model": "cnn", "rows": ["cnn"]}]}
    updated = with_model_block(table, {"model": "transformer", "rows": ["tf"]})
    assert updated == {
        "models": [{"model": "cnn", "rows": ["cnn"]}, {"model": "transformer", "rows": ["tf"]}]
    }


def test_missing_lead_table_has_three_rows_and_matches_the_clean_record():
    assert TABLE_JSON.exists()
    table = json.loads(TABLE_JSON.read_text())
    clean_run = json.loads(METRICS_JSON.read_text())
    block = _model_block(table, "cnn")
    assert [row["condition"] for row in block["rows"]] == [
        "clean",
        "precordial_v1_v3",
        "i_ii_only",
    ]
    assert "average" not in table
    clean = block["rows"][0]
    assert clean["n_test"] == 2198
    assert clean["macro_auroc"] == pytest.approx(clean_run["macro_auroc"])
    assert clean["macro_auprc"] == pytest.approx(clean_run["macro_auprc"])
    for name in SUPERCLASSES:
        assert clean["per_class_auroc"][name] == pytest.approx(
            clean_run["per_class_auroc"][name]
        )
        assert name in clean["per_class_auprc"]
    for row in block["rows"]:
        assert row["n_test"] == 2198
        assert list(row["per_class_auroc"]) == list(SUPERCLASSES)
        assert all(math.isfinite(v) for v in row["per_class_auroc"].values())


def test_demo_shows_three_different_probability_vectors_for_one_fold10_ecg():
    assert DEMO_JSON.exists()
    demo = json.loads(DEMO_JSON.read_text())
    assert isinstance(demo["ecg_id"], int)
    probs = demo["probabilities"]
    assert list(probs) == ["clean", "precordial_v1_v3", "i_ii_only"]
    vectors = []
    for name in probs:
        assert list(probs[name]) == list(SUPERCLASSES)
        vec = tuple(probs[name][c] for c in SUPERCLASSES)
        assert all(0.0 <= p <= 1.0 for p in vec)
        vectors.append(vec)
    assert len(set(vectors)) == 3
    if PACKED_SIGNALS.exists():
        ecg_ids = np.load(PACKED_SIGNALS.parent / "ecg_id.npy")
        folds = np.load(PACKED_SIGNALS.parent / "strat_fold.npy")
        assert int(folds[ecg_ids == demo["ecg_id"]][0]) == 10
