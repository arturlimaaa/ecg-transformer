import json
from pathlib import Path

import torch

from ecg_transformer.cnn import CNN, CONDITIONS, load_split, score_condition, with_model_block
from ecg_transformer.smoke import pick_device

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
CHECKPOINT = ROOT / "checkpoints" / "m5_cnn.pt"
OUT = ROOT / "figures" / "missing_lead_table.json"
BATCH_SIZE = 128


def main() -> None:
    device = pick_device()
    model = CNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    test_x, test_y = load_split(PACKED, "test")
    rows = [
        score_condition(model, test_x, test_y, name, BATCH_SIZE, device)
        for name in CONDITIONS
    ]
    block = {
        "model": "cnn",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "rows": rows,
    }
    old = json.loads(OUT.read_text()) if OUT.exists() else {"models": []}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(with_model_block(old, block), indent=2) + "\n")
    for row in rows:
        print(row["condition"], row["macro_auroc"], row["per_class_auroc"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
