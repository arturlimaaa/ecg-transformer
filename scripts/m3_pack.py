from pathlib import Path

import numpy as np

from ecg_transformer.dataset import PTBXLDataset, pack_normalized_corpus
from ecg_transformer.ptbxl import load_metadata, official_split

ROOT = Path(__file__).resolve().parents[1]
PTBXL = ROOT / "data" / "ptb-xl"
PACKED = PTBXL / "packed"


def main() -> None:
    if (PTBXL / "records500").exists():
        raise SystemExit("records500 appeared on disk")
    packed = pack_normalized_corpus(PTBXL, PACKED)
    meta, labels = load_metadata(PTBXL)
    train = PTBXLDataset(packed, "train")
    val = PTBXLDataset(packed, "val")
    test = PTBXLDataset(packed, "test")
    empty = int((labels.to_numpy().sum(axis=1) == 0).sum())
    print("n", len(meta))
    print("train", len(train), "val", len(val), "test", len(test))
    print("official", {k: len(official_split(meta, k)) for k in ("train", "val", "test")})
    print("empty_labels", empty)
    print("lead_mean", np.load(packed / "lead_mean.npy").tolist())
    print("wrote", packed / "signals.npy")


if __name__ == "__main__":
    main()
