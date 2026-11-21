"""Adapt a STEAD CNN to labelled CWA strong-motion traces before QSIS fine-tuning."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, detrend, resample_poly, sosfilt
from scipy.signal.windows import tukey
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from seismictorch.models.waveform import EarthquakeCNN
from seismictorch.training.pilot import augment_batch, window
from seismictorch.training.runtime import run_epoch, select_device

RATE = 200
FILTER = butter(4, 0.5, fs=RATE, btype="highpass", output="sos")


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def clips(waveforms, metadata):
    """Create onset clips and pre-P background from single-event acceleration."""
    with h5py.File(waveforms) as archive, metadata.open(newline="") as file:
        layout = archive["data_format"]
        if (layout["component_order"][()] != b"ZNE" or layout["sampling_rate"][()] != RATE
                or layout["measurement"][()] != b"acceleration"):
            raise ValueError("Expected the 200 Hz ZNE TSMIP acceleration archive")
        for row in csv.DictReader(file):
            if (row["trace_completeness"] != "4" or row["trace_sampling_rate_hz"] != str(RATE)
                    or row["trace_event_number"] != "1"
                    or any(row[f"trace_{phase}_status"].lower() not in {"maunal", "manual"}
                           for phase in "ps")):
                continue
            try:
                p = float(row["trace_p_arrival_sample"])
                s = float(row["trace_s_arrival_sample"])
            except ValueError:
                continue
            name = row["trace_name"]
            source = np.asarray(archive["data"][name])[[2, 1, 0]]  # E, N, Z
            if not (source.shape[1] == int(row["trace_npts"])
                    and 5 * RATE <= p and p + 2 * RATE <= s < source.shape[1] - RATE
                    and np.isfinite(source).all() and (np.ptp(source, axis=1) > 0).all()):
                continue
            tapered = detrend(source, axis=1) * tukey(source.shape[1], 0.1)
            filtered = resample_poly(sosfilt(FILTER, tapered, axis=1), 1, 2, axis=1)
            raw = source[:, ::2]
            examples = [window(raw, filtered, round(time / 2)) for time in (p, s, p - 3 * RATE)]
            if all(x is not None for x in examples):
                yield row["source_event_id"], name, examples


def run(args):
    if args.output.exists():
        raise ValueError("Use a new output directory")
    examples, groups = [], []
    for event, name, samples in clips(args.waveforms, args.metadata):
        if not event:
            raise ValueError(f"No event ID for {name}")
        examples.extend(samples)
        groups.extend([event] * 3)
    unique = sorted(set(groups))
    if len(unique) < 10:
        raise ValueError("Need at least ten CWA earthquake groups")
    rng = np.random.default_rng(args.seed)
    validation_groups = set(rng.permutation(unique)[:len(unique) // 5])
    inputs = torch.from_numpy(np.stack(examples))
    traces = len(examples) // 3
    labels = torch.arange(3).repeat(traces)
    holdout = torch.tensor([group in validation_groups for group in groups])
    train = DataLoader(TensorDataset(inputs[~holdout], labels[~holdout]), batch_size=128,
                       shuffle=True, collate_fn=augment_batch)
    validation = DataLoader(TensorDataset(inputs[holdout], labels[holdout]), batch_size=128)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model = EarthquakeCNN().to(device)
    model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best = None
    history = []
    args.output.mkdir(parents=True)
    checkpoint = args.output / "best.pth"
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train, device, criterion, optimizer)
        result = run_epoch(model, validation, device, criterion)
        record = {"epoch": epoch, "train": training, "validation": result}
        history.append(record)
        if best is None or result["balanced_accuracy"] > best["validation"]["balanced_accuracy"]:
            best = record
            pending = checkpoint.with_suffix(".tmp")
            torch.save({key: value.cpu() for key, value in model.state_dict().items()}, pending)
            pending.replace(checkpoint)
        print(f"{epoch}/{args.epochs} on {device}: CWA validation {result['balanced_accuracy']:.2%}", flush=True)
    result = {"device": str(device), "seed": args.seed, "best": best, "history": history,
              "clips_by_class": {i: traces for i in range(3)},
              "validation_events": sorted(validation_groups),
              "training_events": sorted(set(unique) - validation_groups),
              "source_sha256": {name: checksum(getattr(args, name))
                                for name in ("metadata", "waveforms", "init")},
              "checkpoint_sha256": checksum(checkpoint)}
    with (args.output / "metrics.json").open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "waveforms", "init", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Specify at least one epoch")
    run(args)
