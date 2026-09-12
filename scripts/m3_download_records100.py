from pathlib import Path

from ecg_transformer.ptbxl import download_records100, load_metadata

ROOT = Path(__file__).resolve().parents[1] / "data" / "ptb-xl"


def main() -> None:
    meta, _ = load_metadata(ROOT)
    names = meta["filename_lr"].tolist()
    if any("records500" in name for name in names):
        raise SystemExit("metadata listed a records500 path")
    download_records100(names, ROOT, max_workers=24)
    missing = [name for name in names if not (ROOT / f"{name}.dat").exists()]
    n_dat = len(list((ROOT / "records100").rglob("*.dat")))
    print("n_meta", len(names))
    print("n_dat", n_dat)
    print("missing", len(missing))
    if missing:
        raise SystemExit(f"missing {len(missing)} records100 .dat files")
    if (ROOT / "records500").exists():
        raise SystemExit("records500 appeared on disk")


if __name__ == "__main__":
    main()
