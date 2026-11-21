"""Diagnose a fixed QSIS checkpoint on public P/S picks; no noise references."""

import argparse
import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
import torch

from seismictorch.models.waveform import WAVEFORM_MODELS
from seismictorch.common import digest, save_json
from seismictorch.qsis.public_data import SHIFTS, public_samples
from seismictorch.data.archives import CLASS_NAMES

def score(args):
    split, labels, training = [
        json.loads(path.read_text()) for path in (args.split, args.labels, args.candidate / "training.json")
    ]
    checkpoint = args.candidate / "model.pth"
    hashes = {name: digest(getattr(args, name))
              for name in ("split", "download", "manifest", "labels")}
    hashes["checkpoint"] = digest(checkpoint)
    if training["split_sha256"] != hashes["split"] or training["checkpoint_sha256"] != hashes["checkpoint"]:
        raise ValueError("The fixed checkpoint does not match the split or training record")

    samples, coverage, waveform_hashes, records = public_samples(
        split, args.download, args.manifest, args.labels
    )
    training_origins = [UTCDateTime(row["origin_utc"]) for row in training["sources"]
                        if "origin_utc" in row and row["label"] != 2]
    if any(abs(UTCDateTime(row["origin_utc"]) - origin) < 120
           for row in records.values() for origin in training_origins):
        raise ValueError("Public evaluation overlaps a training earthquake")

    model = WAVEFORM_MODELS[training["model"]]()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    predictions = []
    with torch.inference_mode():
        for clips, row in samples:
            preds = model(torch.from_numpy(np.stack(clips))).argmax(1).tolist()
            predictions.append({
                "id": row["id"], "event_key": row["event_key"], "station": row["station"],
                "reference": CLASS_NAMES[row["label"]], "time_seconds": row["onset_center"],
                "predicted": [CLASS_NAMES[k] for k in preds],
            })
    events = {}
    for item in predictions:
        events.setdefault(item["event_key"], []).append(item)
    matrices = {}
    for shift_idx, shift in enumerate(SHIFTS):
        counts = np.zeros((2, 3), dtype=int)
        for item in predictions:
            counts[CLASS_NAMES.index(item["reference"]),
                   CLASS_NAMES.index(item["predicted"][shift_idx])] += 1
        matrices[str(shift / 100)] = counts.tolist()
    save_json(args.output, {
        "status": "exploratory_phase_only_audit",
        "label_source": labels["label_source"],
        "limitation": "AI consensus on selected public station traces; no reference noise class.",
        "sha256": hashes,
        "waveform_sha256": waveform_hashes,
        "reference_classes": CLASS_NAMES[:2],
        "predicted_classes": CLASS_NAMES,
        "reviewed_stations": len(records),
        "reviewed_events": len({r["event_key"] for r in records.values()}),
        "coverage": dict(coverage),
        "center_shifts_seconds": [s / 100 for s in SHIFTS],
        "confusion_by_center_shift": matrices,
        "predictions_by_event": events,
    })
    print(f"{len(predictions)} reviewed phase clips across {len(events)} events")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "download", "manifest", "labels", "candidate", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    score(parser.parse_args())
