import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ecg_transformer.cnn import CNN, fit_cnn, load_split, metric_record, predict_logits
from ecg_transformer.smoke import pick_device

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
OUT = ROOT / "figures" / "m5_metrics.json"
CURVES = ROOT / "figures" / "m5_curves.png"
CHECKPOINT = ROOT / "checkpoints" / "m5_cnn.pt"

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
    )
    del train_x, train_y, val_x, val_y

    model = CNN().to(device)
    model.load_state_dict(fitted["state_dict"])
    test_x, test_y = load_split(PACKED, "test")
    scores = predict_logits(model, test_x, BATCH_SIZE, device)
    record = metric_record(test_y.numpy(), scores)
    record["condition"] = "clean"
    record["seed"] = SEED
    record["selected_epoch"] = fitted["selected_epoch"]
    record["selected_val_macro_auroc"] = fitted["selected_val_macro_auroc"]
    record["optimizer"] = "Adam"
    record["lr"] = LR
    record["batch_size"] = BATCH_SIZE
    record["max_epochs"] = MAX_EPOCHS
    record["patience"] = PATIENCE
    record["checkpoint"] = str(CHECKPOINT.relative_to(ROOT))
    record["history"] = fitted["history"]

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fitted["state_dict"], CHECKPOINT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    _plot(fitted["history"], CURVES)
    print(json.dumps({k: record[k] for k in ("n_test", "macro_auroc", "macro_auprc", "per_class_auroc", "selected_epoch")}, indent=2))
    print("wrote", OUT)
    if record["macro_auroc"] <= 0.5:
        raise SystemExit(f"macro-AUROC {record['macro_auroc']} did not beat 0.5")


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
