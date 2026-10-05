import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ecg_transformer.cnn import CNN, CONDITIONS, apply_condition, load_split, predict_logits
from ecg_transformer.dataset import PTBXLDataset
from ecg_transformer.ptbxl import LEADS, SUPERCLASSES
from ecg_transformer.smoke import pick_device

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
FIGURES = ROOT / "figures"
OUT = FIGURES / "slides"
SHIFT = FIGURES / "m9_score_shift.json"

# Matches the deck: paper background, ink text, CNN blue, Transformer orange,
# and an ordinal blue ramp for clean -> V1-V3 off -> I+II only (validated on the paper).
PAPER = "#f7f6f2"
INK = "#17212b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OFF = "#b9b6ac"
CNN_BLUE = "#2a78d6"
TF_ORANGE = "#eb6834"
CONDITION_SHADES = {"clean": "#0d366b", "precordial_v1_v3": "#256abf", "i_ii_only": "#6da7ec"}
CONDITION_LABELS = {"clean": "All 12 leads", "precordial_v1_v3": "V1–V3 off", "i_ii_only": "I + II only"}
# Figures are sized in slide pixels at 100 dpi and saved at 2x.
DPI = 100
plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 19,
        "text.color": INK,
        "axes.labelcolor": SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = {b["model"]: b["rows"] for b in json.loads((FIGURES / "missing_lead_table.json").read_text())["models"]}
    demo = json.loads((FIGURES / "m8_demo.json").read_text())
    _prevalence(json.loads((FIGURES / "m4_metrics.json").read_text())["per_class_train_prevalence"])
    _training(
        json.loads((FIGURES / "m5_metrics.json").read_text()),
        json.loads((FIGURES / "m6_metrics.json").read_text()),
    )
    _macro(table)
    _per_class(table["cnn"])
    _demo(demo)
    _conditions(demo["ecg_id"])
    shift = _score_shift()
    SHIFT.write_text(json.dumps(shift, indent=2) + "\n")
    print("i_ii_only MI negatives above 0.5:", shift["i_ii_only"]["MI"]["negatives_above_half"])
    print("wrote", OUT, SHIFT)


def _figure(width_px: int, height_px: int):
    return plt.subplots(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=2 * DPI, facecolor=PAPER)
    plt.close(fig)


def _hairline_grid(ax, axis: str) -> None:
    ax.grid(axis=axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _prevalence(prevalence: dict[str, float]) -> None:
    fig, ax = _figure(760, 440)
    names = list(SUPERCLASSES)[::-1]
    values = [prevalence[n] for n in names]
    ax.barh(names, values, height=0.45, color=CNN_BLUE)
    for y, v in enumerate(values):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=INK)
    ax.set_xlim(0, 0.55)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of training ECGs with the label")
    ax.tick_params(axis="y", length=0)
    _hairline_grid(ax, "x")
    fig.tight_layout()
    _save(fig, "prevalence.png")


def _training(cnn: dict, tf: dict) -> None:
    fig, ax = _figure(760, 480)
    for run, color, label in ((cnn, CNN_BLUE, "CNN"), (tf, TF_ORANGE, "Transformer")):
        epochs = [h["epoch"] for h in run["history"]]
        aucs = [h["val_macro_auroc"] for h in run["history"]]
        ax.plot(epochs, aucs, color=color, linewidth=2, label=label)
        ax.plot(
            run["selected_epoch"], run["selected_val_macro_auroc"], "o",
            color=color, markersize=10, markeredgecolor=PAPER, markeredgewidth=2,
        )
        ax.text(epochs[-1] + 0.5, aucs[-1], label, va="center", color=INK)
    ax.set_xlim(0, 31)
    ax.set_ylim(0.85, 0.935)
    ax.set_xlabel("epoch")
    ax.set_ylabel("fold-9 macro-AUROC")
    _hairline_grid(ax, "y")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "training.png")


def _macro(table: dict[str, list[dict]]) -> None:
    fig, ax = _figure(1000, 600)
    x = np.arange(len(CONDITIONS))
    ax.axhline(0.5, color=SECONDARY, linewidth=1)
    ax.text(-0.25, 0.51, "chance 0.50", va="bottom", color=SECONDARY)
    for model, color, label, dy in (("cnn", CNN_BLUE, "CNN", 14), ("transformer", TF_ORANGE, "Transformer", -26)):
        values = [row["macro_auroc"] for row in table[model]]
        ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=10,
                markeredgecolor=PAPER, markeredgewidth=2, label=label)
        for xi, v in zip(x, values, strict=True):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, dy), textcoords="offset points", ha="right" if dy < 0 else "center", color=INK)
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("fold-10 macro-AUROC")
    ax.tick_params(axis="x", length=0)
    _hairline_grid(ax, "y")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "macro.png")


def _per_class(rows: list[dict]) -> None:
    by_condition = {row["condition"]: row["per_class_auroc"] for row in rows}
    drop = {c: by_condition["clean"][c] - by_condition["precordial_v1_v3"][c] for c in SUPERCLASSES}
    names = sorted(SUPERCLASSES, key=lambda c: drop[c])
    fig, ax = _figure(1000, 560)
    for y, name in enumerate(names):
        values = [by_condition[c][name] for c in CONDITIONS]
        ax.plot([min(values), max(values)], [y, y], color=AXIS, linewidth=2, zorder=1)
    for condition in CONDITIONS:
        ax.scatter(
            [by_condition[condition][n] for n in names], range(len(names)), s=150,
            color=CONDITION_SHADES[condition], edgecolors=PAPER, linewidths=2,
            label=CONDITION_LABELS[condition], zorder=2,
        )
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("CNN fold-10 AUROC (0.5 = chance)")
    ax.tick_params(axis="y", length=0)
    _hairline_grid(ax, "x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    _save(fig, "per_class.png")


def _demo(demo: dict) -> None:
    fig, ax = _figure(1000, 520)
    x = np.arange(len(SUPERCLASSES))
    width = 0.24
    for k, condition in enumerate(CONDITIONS):
        values = [demo["probabilities"][condition][c] for c in SUPERCLASSES]
        ax.bar(x + (k - 1) * (width + 0.02), values, width, color=CONDITION_SHADES[condition],
               label=CONDITION_LABELS[condition])
    ax.set_xticks(x, SUPERCLASSES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("CNN probability")
    ax.tick_params(axis="x", length=0)
    _hairline_grid(ax, "y")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "demo.png")


def _conditions(ecg_id: int) -> None:
    ds = PTBXLDataset(PACKED, "test")
    j = int(ds._index[np.flatnonzero(ds.ecg_ids[ds._index] == ecg_id)[0]])
    signal = torch.from_numpy(np.array(ds.signals[j], dtype=np.float32, copy=True)).unsqueeze(0)
    seconds = np.arange(400) / 100
    names = [name.replace("AV", "aV") for name in LEADS]
    for condition in CONDITIONS:
        masked = apply_condition(signal, condition)[0, :, :400].numpy()
        fig, ax = _figure(540, 600)
        for k, trace in enumerate(masked):
            off = not trace.any()
            ax.plot(seconds, np.clip(trace, -3, 3) * 0.15 - k, color=OFF if off else INK, linewidth=1.2)
            ax.text(-0.08, -k, names[k], ha="right", va="center", fontsize=17,
                    color=SECONDARY if off else INK)
        ax.set_xlim(-0.55, 4)
        ax.set_ylim(-11.6, 0.6)
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        _save(fig, f"conditions_{condition}.png")


def _score_shift() -> dict:
    device = pick_device()
    model = CNN().to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "m5_cnn.pt", map_location="cpu", weights_only=True))
    test_x, test_y = load_split(PACKED, "test")
    labels = test_y.numpy().astype(bool)
    shift = {}
    for condition in CONDITIONS:
        probs = 1 / (1 + np.exp(-predict_logits(model, apply_condition(test_x, condition), 256, device)))
        shift[condition] = {
            name: {
                "mean_on_positives": float(probs[labels[:, k], k].mean()),
                "mean_on_negatives": float(probs[~labels[:, k], k].mean()),
                "negatives_above_half": float((probs[~labels[:, k], k] > 0.5).mean()),
            }
            for k, name in enumerate(SUPERCLASSES)
        }
    return shift


if __name__ == "__main__":
    main()
