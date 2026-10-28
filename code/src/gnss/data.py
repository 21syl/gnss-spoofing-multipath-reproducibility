"""Raw-epoch loading, boundary-safe windows and training-only preprocessing.

The MATLAB exporter writes /I as [correlators, epochs] in MATLAB, which h5py
reads as [epochs, correlators]. No statistics are fitted during export.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Record:
    name: str
    values: np.ndarray
    state: np.ndarray
    metadata: dict
    digest: str


def load_record(path):
    with h5py.File(path, "r") as handle:
        raw = handle.attrs["Metadata"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        meta = json.loads(raw)
        if meta.get("SchemaVersion") != 4 or meta.get("AmplitudeNormalization") != "none":
            raise ValueError("Expected raw-epoch HDF5 schema 4 from export_correlator_records.m")
        values = np.asarray(handle["I"], dtype=np.float32)
        state = np.asarray(handle["physical_state"]).reshape(-1).astype(np.int64)
    if values.shape != (len(state), meta["Correlators"]):
        raise ValueError("Expected /I with h5py layout [epochs, correlators]")
    if not len(state) or not np.isfinite(values).all() or not np.isin(state, [0, 1, 2, 3]).all():
        raise ValueError("Empty, nonfinite, or invalid-state recording")
    if meta["EpochInterval_ms"] != 1:
        raise ValueError("This configuration uses 1 ms epochs")
    return Record(meta["RecordID"], np.abs(values), state, meta, sha256(path))


def intervals(state):
    edges = np.r_[0, np.flatnonzero(np.diff(state)) + 1, len(state)]
    return list(zip(edges[:-1], edges[1:]))


def windows(record, mode, ratios=(.6, .2, .2), width=10, step=10):
    """Split each raw state interval first; never form a boundary-crossing window."""
    if width != 10 or step != 10:
        raise ValueError("Paper protocol uses 10 ms nonoverlapping windows")
    if not np.isclose(sum(ratios), 1) or min(ratios) <= 0:
        raise ValueError("Split ratios must be positive and sum to one")
    chunks, rows = [], []
    for block, (start, stop) in enumerate(intervals(record.state)):
        if mode == "chronological":
            # Integer epoch boundaries are applied before window construction.
            n = stop - start
            boundaries = [start, start + int(n * ratios[0]),
                          start + int(n * (ratios[0] + ratios[1])), stop]
            segments = list(zip(boundaries[:-1], boundaries[1:], ["train", "validation", "test"]))
        elif mode == "unassigned":
            segments = [(start, stop, "unassigned")]
        else:
            raise ValueError(f"Unknown window mode {mode}")
        for left, right, part in segments:
            for first in range(left, right - width + 1, step):
                state = int(record.state[first])
                assert np.all(record.state[first:first + width] == state)
                label = 1 if state in (1, 3) else (2 if state == 2 else 0)
                chunks.append(record.values[first:first + width].T)
                rows.append({"record": record.name, "start_ms": first, "end_ms": first + width,
                             "physical_state": state, "label": label, "partition": part,
                             "state_block": block})
    if not chunks:
        raise ValueError(f"No complete windows in {record.name}")
    return np.stack(chunks), pd.DataFrame(rows)


def random_partition(frame, ratios, seed, names):
    ids = np.arange(len(frame))
    if len(ratios) != len(names) or not np.isclose(sum(ratios), 1) or min(ratios) <= 0:
        raise ValueError("Invalid random split ratios")
    first, remaining = train_test_split(ids, train_size=ratios[0],
                                        stratify=frame.label, random_state=seed)
    assignment = np.empty(len(ids), dtype=object)
    assignment[first] = names[0]
    if len(ratios) == 2:
        assignment[remaining] = names[1]
    else:
        second, third = train_test_split(remaining, train_size=ratios[1] / sum(ratios[1:]),
                                         stratify=frame.label.iloc[remaining], random_state=seed)
        assignment[second], assignment[third] = names[1], names[2]
    frame = frame.copy()
    frame["partition"] = assignment
    return frame


def balance_partitions(frame, seed):
    """Downsample within each partition, without augmentation or crossing splits."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in frame.groupby("partition", sort=True):
        size = group.groupby("label").size().min()
        for _, cls in group.groupby("label", sort=True):
            keep.extend(rng.choice(cls.index, int(size), replace=False).tolist())
    return np.array(sorted(keep), dtype=int)


def prepare(config, data_dir):
    if config['protocol'] not in ('within_chronological', 'independent_record_transfer'):
        raise ValueError('Unknown data-partition protocol')
    if config['protocol'] == 'within_chronological' and len(config['split_ratios']) != 3:
        raise ValueError('Chronological evaluation requires train/validation/test ratios')
    ids = config["source_records"] + config.get("target_records", [])
    if len(ids) != len(set(ids)):
        raise ValueError("Source and target IDs must be unique and disjoint")
    records = {name: load_record(Path(data_dir) / f"{name}.h5") for name in ids}
    if any(name != rec.name for name, rec in records.items()):
        raise ValueError("Filename and RecordID disagree")
    if len({rec.digest for rec in records.values()}) != len(records):
        raise ValueError("Duplicate source/target file content")
    reference = next(iter(records.values())).metadata
    for rec in records.values():
        for key in ("Correlators", "SelectedIFields", "SelectedCorrelatorOffsets_chips", "EpochInterval_ms"):
            if rec.metadata[key] != reference[key]:
                raise ValueError(f"Input configuration mismatch: {key}")
    source_x, source_rows = [], []
    chronological = config["protocol"] == "within_chronological"
    for name in config["source_records"]:
        x, rows = windows(records[name], "chronological" if chronological else "unassigned",
                          config.get("split_ratios", [.6, .2, .2]))
        source_x.append(x); source_rows.append(rows)
    x = np.concatenate(source_x)
    rows = pd.concat(source_rows, ignore_index=True)
    if not chronological:
        ratios = config["split_ratios"]
        names = ["train", "validation"] if len(ratios) == 2 else ["train", "validation", "source_holdout"]
        rows = random_partition(rows, ratios, config["seed"], names)
    rows["selected"] = True
    if config.get("balance_source_classes", False):
        keep = balance_partitions(rows, config["seed"])
        rows["selected"] = False
        rows.loc[keep, "selected"] = True
    all_x, all_rows = [x], [rows]
    for name in config.get("target_records", []):
        target_x, target_rows = windows(records[name], "unassigned")
        target_rows["partition"] = "target_test"
        target_rows["selected"] = True
        all_x.append(target_x); all_rows.append(target_rows)
    x, rows = np.concatenate(all_x), pd.concat(all_rows, ignore_index=True)
    rows["sample_id"] = np.arange(len(rows))
    training = (rows.partition == "train") & rows.selected
    if not training.any():
        raise ValueError("No training samples")
    scaler = Scaler.fit(x[training], config["normalization"])
    return scaler.transform(x), rows, scaler, {name: {"sha256": r.digest, "metadata": r.metadata}
                                               for name, r in records.items()}


@dataclass
class Scaler:
    method: str
    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training, method):
        if method == "train_zscore_per_correlator":
            offset = training.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
            scale = training.std(axis=(0, 2), keepdims=True, dtype=np.float64)
        elif method == "train_global_minmax":
            offset = np.asarray(training.min()).reshape(1, 1, 1)
            scale = np.asarray(training.max() - training.min()).reshape(1, 1, 1)
        else:
            raise ValueError(f"Unsupported normalization {method}")
        return cls(method, offset, np.where(scale > 1e-12, scale, 1.0))

    def transform(self, x):
        # Test values can lie outside [0,1]. Do not refit or clip them.
        return np.ascontiguousarray((x - self.offset) / self.scale, dtype=np.float32)

    def to_dict(self):
        return {"method": self.method, "fit_partition": "selected source training samples only",
                "offset": self.offset.tolist(), "scale": self.scale.tolist(), "clip": False}
