from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecg_transformer.baseline import ranking_metrics
from ecg_transformer.dataset import PTBXLDataset
from ecg_transformer.ptbxl import LEADS, SUPERCLASSES

CONDITIONS = ("clean", "precordial_v1_v3", "i_ii_only")
_ZERO_LEADS = {
    "clean": (),
    "precordial_v1_v3": ("V1", "V2", "V3"),
    "i_ii_only": tuple(name for name in LEADS if name not in {"I", "II"}),
}


class CNN(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def load_split(packed_dir: Path | str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    ds = PTBXLDataset(packed_dir, split)
    signals = np.array(ds.signals[ds._index], dtype=np.float32, copy=True)
    labels = np.array(ds.labels[ds._index], dtype=np.float32, copy=True)
    return torch.from_numpy(signals), torch.from_numpy(labels)


def apply_condition(signals: torch.Tensor, condition: str) -> torch.Tensor:
    try:
        names = _ZERO_LEADS[condition]
    except KeyError as err:
        raise ValueError(f"unknown condition {condition!r}") from err
    out = signals.clone()
    if names:
        out[:, [LEADS.index(name) for name in names], :] = 0
    return out


def metric_record(labels: np.ndarray, scores: np.ndarray) -> dict:
    rank = ranking_metrics(labels, scores)
    return {
        "n_test": int(len(labels)),
        "per_class_auroc": {
            name: auc
            for name, auc in zip(SUPERCLASSES, rank["per_class_auroc"], strict=True)
        },
        "macro_auroc": rank["macro_auroc"],
        "per_class_auprc": {
            name: ap
            for name, ap in zip(SUPERCLASSES, rank["per_class_auprc"], strict=True)
        },
        "macro_auprc": rank["macro_auprc"],
    }


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    signals: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    pieces: list[torch.Tensor] = []
    for start in range(0, len(signals), batch_size):
        batch = signals[start : start + batch_size].to(device)
        pieces.append(model(batch).cpu())
    return torch.cat(pieces).numpy()


def score_condition(
    model: nn.Module,
    signals: torch.Tensor,
    labels: torch.Tensor,
    condition: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    masked = apply_condition(signals, condition)
    record = metric_record(labels.numpy(), predict_logits(model, masked, batch_size, device))
    record["condition"] = condition
    return record


def fit_cnn(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    seed: int,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    build_model: Callable[[], nn.Module] = CNN,
) -> dict:
    """Train on the given tensors. Selection uses validation macro-AUROC only."""
    torch.manual_seed(seed)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        val_scores = predict_logits(model, val_x, batch_size, device)
        val_auc = float(ranking_metrics(val_y.numpy(), val_scores)["macro_auroc"])
        train_loss = total / len(train_x)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_macro_auroc": val_auc}
        )
        print(
            f"epoch {epoch} train_loss {train_loss:.4f} val_macro_auroc {val_auc:.4f}",
            flush=True,
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("fit produced no checkpoint")
    return {
        "state_dict": best_state,
        "selected_epoch": best_epoch,
        "selected_val_macro_auroc": best_auc,
        "history": history,
    }
