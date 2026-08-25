"""Validate that the expected reproducibility-package structure is present."""

from pathlib import Path


REQUIRED = (
    "README.md",
    "CITATION.cff",
    "environment/environment.yml",
    "configs/self_collected_bds_b1i/demo_train.yaml",
    "configs/texbat_gps_l1/demo_eval.yaml",
    "data/metadata/dataset_manifest_demo.csv",
    "data/splits/self_collected_record_level_split_demo.csv",
    "src/models/transformer_lstm_demo.py",
    "results/metrics/metrics_demo.csv",
    "docs/REPRODUCIBILITY_CHECKLIST.md",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [item for item in REQUIRED if not (root / item).exists()]
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join(missing))
    print(f"Structure check passed for {len(REQUIRED)} required paths.")


if __name__ == "__main__":
    main()
