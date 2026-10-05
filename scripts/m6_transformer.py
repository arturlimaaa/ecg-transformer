import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ecg_transformer.cnn import CONDITIONS, fit_cnn, load_split, score_condition, with_model_block
from ecg_transformer.smoke import pick_device
from ecg_transformer.transformer import (
    DIM_FEEDFORWARD,
    D_MODEL,
    N_LAYERS,
    NHEAD,
    PATCH_SAMPLES,
    Transformer,
)

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
OUT = ROOT / "figures" / "m6_metrics.json"
CURVES = ROOT / "figures" / "m6_curves.png"
CHECKPOINT = ROOT / "checkpoints" / "m6_transformer.pt"
TABLE = ROOT / "figures" / "missing_lead_table.json"

SEED = 0
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 25
PATIENCE = 5


def main() -> None:
    device = pick_device()
    print("device", device, flush=True)
    train_x, train_y = load_split(PACKED, "train")
    val_x, val_y = load_split(PACKED, "val")
    print("train", tuple(train_x.shape), "val", tuple(val_x.shape), flush=True)
    fitted = fit_cnn(
        train_x,
        train_y,
        val_x,
        val_y,
        seed=SEED,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        device=device,
        build_model=Transformer,
    )
    del train_x, train_y, val_x, val_y

    model = Transformer().to(device)
    model.load_state_dict(fitted["state_dict"])
    test_x, test_y = load_split(PACKED, "test")
    rows = [
        score_condition(model, test_x, test_y, name, BATCH_SIZE, device) for name in CONDITIONS
    ]
    record = dict(rows[0])
    record.update(
        {
            "seed": SEED,
            "selected_epoch": fitted["selected_epoch"],
            "selected_val_macro_auroc": fitted["selected_val_macro_auroc"],
            "optimizer": "Adam",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "patch_samples": PATCH_SAMPLES,
            "patch_ms": PATCH_SAMPLES * 10,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "n_layers": N_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "history": fitted["history"],
        }
    )
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted["state_dict"], CHECKPOINT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    _plot(fitted["history"], CURVES)
    _add_to_table(rows)
    print(
        json.dumps(
            {
                "n_test": record["n_test"],
                "macro_auroc": record["macro_auroc"],
                "macro_auprc": record["macro_auprc"],
                "per_class_auroc": record["per_class_auroc"],
                "selected_epoch": record["selected_epoch"],
                "patch_samples": record["patch_samples"],
            },
            indent=2,
        )
    )
    for row in rows:
        print(row["condition"], row["macro_auroc"], row["per_class_auroc"])
    print("wrote", OUT)
    if record["macro_auroc"] <= 0.5:
        raise SystemExit(f"macro-AUROC {record['macro_auroc']} did not beat 0.5")


def _add_to_table(rows: list[dict]) -> None:
    old = json.loads(TABLE.read_text()) if TABLE.exists() else {"models": []}
    block = {
        "model": "transformer",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "rows": rows,
    }
    TABLE.write_text(json.dumps(with_model_block(old, block), indent=2) + "\n")


def _plot(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(epochs, [row["train_loss"] for row in history], color="black")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[1].plot(epochs, [row["val_macro_auroc"] for row in history], color="black")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("fold-9 macro-AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
