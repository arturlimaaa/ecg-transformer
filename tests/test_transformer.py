import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from ecg_transformer.ptbxl import SUPERCLASSES
from ecg_transformer.transformer import PATCH_SAMPLES, Transformer

METRICS_JSON = Path("figures/m6_metrics.json")
TABLE_JSON = Path("figures/missing_lead_table.json")


def test_transformer_maps_batch_to_five_logits():
    logits = Transformer()(torch.zeros(2, 12, 1000))
    assert logits.shape == (2, 5)


def test_transformer_loss_on_one_batch_is_finite_and_falls():
    torch.manual_seed(0)
    model = Transformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.randn(4, 12, 1000)
    y = torch.randint(0, 2, (4, 5)).float()
    losses: list[float] = []
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]


def test_fold10_transformer_json_beats_chance_and_records_patch_length():
    assert METRICS_JSON.exists()
    payload = json.loads(METRICS_JSON.read_text())
    assert payload["condition"] == "clean"
    assert payload["n_test"] == 2198
    assert payload["patch_samples"] == PATCH_SAMPLES
    assert payload["patch_ms"] == PATCH_SAMPLES * 10
    assert list(payload["per_class_auroc"]) == list(SUPERCLASSES)
    assert list(payload["per_class_auprc"]) == list(SUPERCLASSES)
    assert payload["macro_auroc"] > 0.5
    assert all(math.isfinite(v) for v in payload["per_class_auroc"].values())
    history = payload["history"]
    chosen = next(row for row in history if row["epoch"] == payload["selected_epoch"])
    assert chosen["val_macro_auroc"] == max(row["val_macro_auroc"] for row in history)
    assert Path(payload["checkpoint"]).exists()


def test_missing_lead_table_includes_the_transformer_and_matches_its_clean_record():
    assert TABLE_JSON.exists()
    table = json.loads(TABLE_JSON.read_text())
    clean_run = json.loads(METRICS_JSON.read_text())
    block = next(item for item in table["models"] if item["model"] == "transformer")
    assert [row["condition"] for row in block["rows"]] == [
        "clean",
        "precordial_v1_v3",
        "i_ii_only",
    ]
    clean = block["rows"][0]
    assert clean["n_test"] == 2198
    assert clean["macro_auroc"] == pytest.approx(clean_run["macro_auroc"])
    assert clean["macro_auprc"] == pytest.approx(clean_run["macro_auprc"])
    for name in SUPERCLASSES:
        assert clean["per_class_auroc"][name] == pytest.approx(
            clean_run["per_class_auroc"][name]
        )
