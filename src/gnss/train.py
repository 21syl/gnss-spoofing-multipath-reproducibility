"""Config-driven training, immutable split manifests and held-out evaluation.

Run from repository root: python scripts/run_experiment.py --config CONFIG.json
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import platform

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import prepare
from .metrics import stage_results
from .models import build_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batches(x, y, indices, batch_size, seed, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(x[indices]), torch.as_tensor(y[indices], dtype=torch.long)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)


def fit(model, train_loader, validation_loader, cfg, device, destination, smoke=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
                 if cfg["scheduler"] == "step10_gamma0.1" else None)
    if cfg["scheduler"] not in ("step10_gamma0.1", "none"):
        raise ValueError("Unknown scheduler")
    criterion = nn.CrossEntropyLoss()
    best_loss, stalled, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, (1 if smoke else cfg["max_epochs"]) + 1):
        model.train(); total, count = 0.0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward(); optimizer.step()
            total += loss.item() * len(y); count += len(y)
            if smoke and batch >= 1:
                break
        model.eval(); val_loss, val_count, correct = 0.0, 0, 0
        with torch.inference_mode():
            for batch, (x, y) in enumerate(validation_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                val_count += len(y); correct += (logits.argmax(1) == y).sum().item()
                if smoke and batch >= 1:
                    break
        val_loss /= val_count
        history.append({"epoch": epoch, "train_loss": total / count,
                        "validation_loss": val_loss, "validation_accuracy": correct / val_count,
                        "learning_rate": learning_rate})
        print(f"epoch {epoch}: train loss {total/count:.5f}, validation loss {val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss, stalled = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "epoch": epoch,
                        "validation_loss": val_loss, "smoke_test_only": smoke}, destination)
        else:
            stalled += 1
        if scheduler is not None:
            scheduler.step()
        if stalled >= cfg["early_stopping_patience"]:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(destination.with_suffix(".history.csv"), index=False)


def infer(model, loader, device):
    model.eval(); probability = []
    with torch.inference_mode():
        for x, _ in loader:
            probability.append(model(x.to(device)).softmax(1).cpu().numpy())
    result = np.concatenate(probability)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="One epoch/two batches; never paper results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.models:
        cfg["models"] = args.models
    torch.set_num_threads(4)
    seed_all(cfg["seed"])
    x, frame, scaler, inputs = prepare(cfg, args.data_dir)
    y = frame.label.to_numpy()
    if x.shape[1:] != (cfg["correlators"], 10) or y.max() >= cfg["classes"]:
        raise ValueError("Configuration disagrees with input shape or labels")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.output_dir / (("SMOKE_" if args.smoke_test else "") + cfg["experiment"] + "_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    write_json(out / "preprocessing.json", scaler.to_dict())
    write_json(out / "input_manifest.json", inputs)
    write_json(out / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "device": args.device, "smoke_test_only": args.smoke_test,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    frame.to_csv(out / "split_manifest.csv", index=False)
    frame.groupby(["record", "partition", "label", "selected"]).size().rename("samples").to_csv(out / "sample_counts.csv")
    if args.dry_run:
        print(f"Validated data, split and preprocessing: {out}")
        return
    selected = frame.selected.to_numpy()
    train = np.flatnonzero(selected & (frame.partition == "train"))
    validation = np.flatnonzero(selected & (frame.partition == "validation"))
    if not len(train) or not len(validation):
        raise ValueError("Empty train/validation split")
    rows = []
    for name in cfg["models"]:
        seed_all(cfg["seed"])
        model = build_model(name, cfg["correlators"], cfg["classes"]).to(args.device)
        (out / name).mkdir()
        model_dir = out / name
        (model_dir / "architecture.txt").write_text(str(model), encoding="utf-8")
        write_json(model_dir / "parameter_count.json", {"parameters": sum(p.numel() for p in model.parameters())})
        train_loader = batches(x, y, train, cfg["batch_size"], cfg["seed"], True)
        val_loader = batches(x, y, validation, cfg["batch_size"], cfg["seed"])
        fit(model, train_loader, val_loader, cfg, args.device, model_dir / "best.pt", args.smoke_test)
        for (part, record), group in frame[selected & frame.partition.isin(["test", "target_test", "source_holdout"])].groupby(["partition", "record"]):
            ids = group.index.to_numpy()
            probability = infer(model, batches(x, y, ids, cfg["batch_size"], cfg["seed"]), args.device)
            predictions = group.copy()
            predictions["prediction"] = probability.argmax(1)
            for c in range(cfg["classes"]):
                predictions[f"probability_{c}"] = probability[:, c]
            predictions.to_csv(model_dir / f"{part}_{record}.csv", index=False)
            for row in stage_results(y[ids], probability, group.physical_state.to_numpy()):
                rows.append({"model": name, "record": record, "partition": part,
                             "smoke_test_only": args.smoke_test, **row})
        del model
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
