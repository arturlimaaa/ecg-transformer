import json
from pathlib import Path

from ecg_transformer.baseline import majority_metrics

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "ptb-xl" / "packed"
OUT = ROOT / "figures" / "m4_metrics.json"


def main() -> None:
    metrics = majority_metrics(PACKED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
