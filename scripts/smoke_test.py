"""Exercise all five models on synthetic inputs, never manuscript evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="gnss_synthetic_smoke_") as temp:
        folder = Path(temp)
        rng = np.random.default_rng(42)
        for name in ["bs5", "bs6"]:
            with h5py.File(folder / f"{name}.h5", "w") as handle:
                handle["I"] = rng.normal(size=(3000, 11)).astype("float32")
                handle["physical_state"] = np.repeat([0, 2, 3], 1000)
                handle.attrs["Metadata"] = json.dumps({"SchemaVersion": 4, "RecordID": name,
                    "AmplitudeNormalization": "none", "Correlators": 11, "EpochInterval_ms": 1,
                    "SelectedIFields": [f"synthetic_{i}" for i in range(11)],
                    "SelectedCorrelatorOffsets_chips": np.linspace(-2, 2, 11).tolist(),
                    "SyntheticTestFixture": True})
        subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"),
            "--config", str(ROOT / "configs/experiments/mixed_static_to_dynamic.json"),
            "--data-dir", str(folder), "--output-dir", str(ROOT / "outputs/synthetic_validation"),
            "--device", "cpu", "--smoke-test"], check=True)


if __name__ == "__main__":
    main()
