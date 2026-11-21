"""Inference for an existing arrival center using a checkpoint's recorded preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from seismictorch.common import digest
from seismictorch.data.archives import CLASS_NAMES
from seismictorch.data.preprocessing import prepared_waveform, window
from seismictorch.data.sac import load_xyz
from seismictorch.inference.history import describe, predict as predict_with_history
from seismictorch.models.waveform import WAVEFORM_MODELS
from seismictorch.paths import DEFAULT_CANDIDATE


def predict_waveform(waveforms, center_seconds, candidate=DEFAULT_CANDIDATE, history=None):
    candidate = Path(candidate)
    checkpoint = candidate / "model.pth"
    training = json.loads((candidate / "training.json").read_text())
    if digest(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from its training receipt")
    stream = load_xyz([Path(path) for path in waveforms], min_samples=200)
    if stream is None:
        raise ValueError("Expected aligned 100 Hz SAC waveforms in X, Y, Z order")
    if not np.isfinite(np.stack([trace.data for trace in stream])).all():
        raise ValueError("Waveform contains non-finite samples")
    center = round(center_seconds * 100)
    waveform_filter = training.get("waveform_filter", "tapered")
    raw, filtered, orientation = prepared_waveform(stream, center, waveform_filter, training.get("orientation"))
    clip = window(raw, filtered, center)
    if clip is None:
        raise ValueError("The center needs a complete, usable two-second clip")
    model = WAVEFORM_MODELS[training["model"]]().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    features = None
    with torch.inference_mode():
        logits = model(torch.from_numpy(clip[None, ...]))
        index = int(logits.argmax(dim=-1).item())
        original = CLASS_NAMES[index]
        if history:
            rotation = orientation["rotation_xyz"] if orientation is not None else None
            features = describe(raw, center, rotation=rotation)
            head = torch.load(history, weights_only=True, map_location="cpu")
            index = int(predict_with_history(logits, [features], head).item())
    return {
        "label": CLASS_NAMES[index], "local_cnn_label": original,
        "center_seconds_from_start": center / 100, "waveform_filter": waveform_filter,
        "orientation": orientation, "history_available": features is not None,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "history_sha256": digest(Path(history)) if history else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--waveform", type=Path, nargs=3, metavar=("X", "Y", "Z"), required=True)
    parser.add_argument("--center", type=float, required=True, help="Seconds from SAC recording start")
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()
    print(json.dumps(predict_waveform(args.waveform, args.center, args.candidate, args.history), indent=2))


if __name__ == "__main__":
    main()
