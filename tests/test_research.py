"""Implementation tests use synthetic arrays, never manuscript observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss.data import Record, Scaler, windows, random_partition, load_record, prepare
from gnss.metrics import summarize
from gnss.models import build_model, CNN, LSTM, ResNet1D, Transformer


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_table3_architectures(self):
        self.assertEqual(LSTM().recurrent.num_layers, 2)
        self.assertEqual(LSTM().recurrent.hidden_size, 128)
        cnn = CNN()
        self.assertEqual([m.kernel_size for m in cnn.modules() if isinstance(m, torch.nn.Conv1d)], [(3,)] * 3)
        self.assertEqual(sum(isinstance(m, torch.nn.MaxPool1d) for m in cnn.modules()), 3)
        self.assertEqual(len(ResNet1D().blocks), 4)
        transformer = Transformer()
        self.assertEqual(len(transformer.encoder.layers), 4)
        self.assertEqual(transformer.encoder.layers[0].self_attn.num_heads, 8)
        self.assertEqual(transformer.project.out_features, 256)
        self.assertEqual(transformer.encoder.layers[0].linear1.out_features, 512)

    def test_models_forward_backward_and_batch_independence(self):
        for name in ["cnn", "lstm", "resnet1d", "transformer", "proposed",
                     "no_spatial", "no_multiscale", "no_spatial_no_multiscale"]:
            for classes in (2, 3):
                with self.subTest(name=name, classes=classes):
                    model = build_model(name, 11, classes)
                    x = torch.randn(4, 11, 10)
                    logits = model(x)
                    self.assertEqual(tuple(logits.shape), (4, classes))
                    torch.nn.functional.cross_entropy(logits, torch.arange(4) % classes).backward()
                    model.eval()
                    with torch.no_grad():
                        alone = model(x[:1])
                        together = model(x)[:1]
                    torch.testing.assert_close(alone, together, atol=1e-5, rtol=1e-4)

    def test_split_before_windows(self):
        state = np.repeat([0, 2, 3], 103)
        rec = Record("test", np.zeros((309, 11), np.float32), state, {}, "")
        x, rows = windows(rec, "chronological")
        self.assertEqual(x.shape[1:], (11, 10))
        used = set()
        for row in rows.itertuples():
            epochs = set(range(row.start_ms, row.end_ms))
            self.assertFalse(used & epochs)
            self.assertEqual(len(set(state[row.start_ms:row.end_ms])), 1)
            used |= epochs
        for _, block in rows.groupby("state_block"):
            self.assertLessEqual(block[block.partition == "train"].end_ms.max(),
                                 block[block.partition == "validation"].start_ms.min())

    def test_random_split_reproducible(self):
        rec = Record("test", np.zeros((3000, 11), np.float32), np.repeat([0, 2, 3], 1000), {}, "")
        _, rows = windows(rec, "unassigned")
        one = random_partition(rows, [.7, .15, .15], 42, ["train", "validation", "holdout"])
        two = random_partition(rows, [.7, .15, .15], 42, ["train", "validation", "holdout"])
        self.assertTrue(one.equals(two))
        self.assertEqual(one.groupby("partition").size().to_dict(), {"holdout": 45, "train": 210, "validation": 45})

    def test_training_only_scaler(self):
        training = np.arange(220, dtype=np.float32).reshape(2, 11, 10)
        for method in ["train_global_minmax", "train_zscore_per_correlator"]:
            scaler = Scaler.fit(training, method)
            offset = scaler.offset.copy()
            scaler.transform(np.full((1, 11, 10), 1e6))
            np.testing.assert_array_equal(scaler.offset, offset)
        scaler = Scaler.fit(training, "train_global_minmax")
        self.assertGreater(scaler.transform(np.full((1, 11, 10), 1e6)).min(), 1)

    def test_absent_class_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([[.8,.1,.1],[.1,.1,.8],[.1,.8,.1],[.1,.8,.1]])
        report = summarize(y, p, np.array([0, 0, 1, 3]))
        self.assertEqual(report["accuracy"], .75)
        self.assertIsNone(report["macro_auc_all_classes"])
        self.assertEqual(report["mixed_detection_rate"], 1.)

    def test_export_schema_and_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, shift in [("bs5", 0), ("bs6", 100)]:
                with h5py.File(Path(folder) / f"{name}.h5", "w") as h:
                    h["I"] = np.arange(33000, dtype=np.float32).reshape(3000, 11) + shift
                    h["physical_state"] = np.repeat([0, 2, 3], 1000)
                    h.attrs["Metadata"] = json.dumps({"SchemaVersion": 4, "RecordID": name,
                        "AmplitudeNormalization": "none", "Correlators": 11, "EpochInterval_ms": 1,
                        "SelectedIFields": [str(i) for i in range(11)],
                        "SelectedCorrelatorOffsets_chips": list(range(11))})
            cfg = json.loads((ROOT / "configs/experiments/mixed_static_to_dynamic.json").read_text())
            x, rows, scaler, _ = prepare(cfg, folder)
            self.assertEqual(len(rows[rows.partition == "target_test"]), 300)
            self.assertFalse((rows[rows.partition == "train"].record == "bs6").any())
            self.assertEqual(x.shape, (600, 11, 10))
            self.assertEqual(load_record(Path(folder) / 'bs5.h5').name, 'bs5')


if __name__ == "__main__":
    unittest.main()
