import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ecg_transformer.cnn import CNN, CONDITIONS, apply_condition, predict_logits
from ecg_transformer.dataset import PTBXLDataset
from ecg_transformer.ptbxl import SUPERCLASSES
from ecg_transformer.smoke import pick_device

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
CHECKPOINT = ROOT / "checkpoints" / "m5_cnn.pt"
OUT = ROOT / "figures" / "m8_demo.json"
FIG = ROOT / "figures" / "m8_demo.png"
LABELS = {
    "clean": "all 12 leads",
    "precordial_v1_v3": "V1–V3 off",
    "i_ii_only": "I+II only",
}


def main() -> None:
    device = pick_device()
    ds = PTBXLDataset(PACKED, "test")
    labels = np.array(ds.labels[ds._index], dtype=np.float32)
    mi = SUPERCLASSES.index("MI")
    chosen = np.flatnonzero((labels[:, mi] == 1) & (labels.sum(axis=1) == 1))
    if chosen.size == 0:
        raise SystemExit("no fold-10 record labeled only MI")
    model = CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    pool = torch.from_numpy(np.array(ds.signals[ds._index[chosen]], dtype=np.float32, copy=True))
    clean_mi = torch.sigmoid(
        torch.from_numpy(predict_logits(model, pool, 128, device)[:, mi])
    ).numpy()
    agreed = np.flatnonzero(clean_mi >= 0.8)
    if agreed.size == 0:
        raise SystemExit("no MI-only fold-10 record with clean MI probability >= 0.8")
    i = int(chosen[int(agreed[0])])
    j = int(ds._index[i])
    signal = torch.from_numpy(np.array(ds.signals[j], dtype=np.float32, copy=True))
    probabilities = {
        name: _probs(model, signal, name, device) for name in CONDITIONS
    }
    vectors = [tuple(probabilities[name][c] for c in SUPERCLASSES) for name in CONDITIONS]
    if len(set(vectors)) != 3:
        raise SystemExit("demo probabilities did not change across masks")
    demo = {
        "ecg_id": int(ds.ecg_ids[j]),
        "patient_id": int(ds.patient_ids[j]),
        "selection": "first fold-10 MI-only record with clean MI probability >= 0.8",
        "label": {name: float(labels[i, k]) for k, name in enumerate(SUPERCLASSES)},
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "probabilities": probabilities,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(demo, indent=2) + "\n")
    _plot(demo, FIG)
    print("ecg_id", demo["ecg_id"])
    for name in CONDITIONS:
        print(name, probabilities[name])
    print("wrote", OUT)


def _probs(model: CNN, signal: torch.Tensor, condition: str, device: torch.device) -> dict[str, float]:
    logits = predict_logits(model, apply_condition(signal.unsqueeze(0), condition), 1, device)[0]
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return {name: float(p) for name, p in zip(SUPERCLASSES, probs, strict=True)}


def _plot(demo: dict, path: Path) -> None:
    x = np.arange(len(SUPERCLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, name in enumerate(CONDITIONS):
        vals = [demo["probabilities"][name][c] for c in SUPERCLASSES]
        ax.bar(x + (k - 1) * width, vals, width, label=LABELS[name])
    ax.set_xticks(x, SUPERCLASSES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability")
    ax.set_title(f"ecg {demo['ecg_id']}, MI only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
