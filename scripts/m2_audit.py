from pathlib import Path

import matplotlib.pyplot as plt

from ecg_transformer.ptbxl import (
    SUPERCLASSES,
    download_records100,
    load_metadata,
    load_record_100hz,
    patient_fold_overlap_count,
    select_debug_records,
)

ROOT = Path(__file__).resolve().parents[1]
PTBXL = ROOT / "data" / "ptb-xl"
FIGURES = ROOT / "figures"


def class_names(row) -> str:
    names = [name for name in SUPERCLASSES if row[name] == 1]
    return "+".join(names) if names else "none"


def main() -> None:
    meta, labels = load_metadata(PTBXL)
    overlap = patient_fold_overlap_count(meta)
    if overlap != 0:
        raise SystemExit(f"patient overlap across folds: {overlap}")
    chosen = select_debug_records(meta, labels, n=100, seed=0)
    download_records100(chosen["filename_lr"], PTBXL)
    signals = [load_record_100hz(PTBXL / rel) for rel in chosen["filename_lr"]]
    if any(sig.shape != (12, 1000) for sig in signals):
        raise SystemExit("a 100 Hz record was not shape (12, 1000)")
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(10, 10, figsize=(20, 16), sharex=True, sharey=True)
    chosen_y = labels.loc[chosen.index]
    for ax, sig, (_, row), (_, yrow) in zip(
        axes.ravel(), signals, chosen.iterrows(), chosen_y.iterrows(), strict=True
    ):
        ax.plot(sig[1], color="black", linewidth=0.4)
        ax.set_title(f"{int(row.ecg_id)} {class_names(yrow)}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("M2 audit: 100 PTB-XL 100 Hz records, lead II")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "m2_lead2_grid.png", dpi=120)
    plt.close(fig)
    out = chosen[["ecg_id", "patient_id", "strat_fold", "filename_lr"]].copy()
    out["patient_id"] = out["patient_id"].astype(int)
    for name in SUPERCLASSES:
        out[name] = chosen_y[name].to_numpy()
    out.to_csv(FIGURES / "m2_labels.csv", index=False)
    print("overlap", overlap)
    print("n", len(chosen))
    print("class_counts", {name: int(chosen_y[name].sum()) for name in SUPERCLASSES})
    print("repeat_patients", int((chosen.groupby("patient_id").size() >= 2).sum()))
    print("cooccur", int((chosen_y.sum(axis=1) >= 2).sum()))
    print("wrote", FIGURES / "m2_lead2_grid.png")


if __name__ == "__main__":
    main()
