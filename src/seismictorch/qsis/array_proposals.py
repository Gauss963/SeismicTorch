"""Expand training picks with agreement across nearby QSIS instruments and PhaseNet."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from obspy import read
import torch

from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.training.runtime import select_device

RATE = 100
GAP_SECONDS = 0.5


def site(row):
    sac = read(str(ROOT / row["paths"][0]), format="SAC", headonly=True)[0].stats.sac
    return round(float(sac.stla), 3), round(float(sac.stlo), 3)


def clustered_time(picks):
    """Require at least three distinct stations and reject a competing packet."""
    times = np.sort(list(picks))
    runs = [times[(times >= t) & (times <= t + GAP_SECONDS)] for t in times]
    if not runs:
        return None
    largest = max(runs, key=len)
    if len(largest) < 3 or any(len(run) >= len(largest) - 1 and
                                abs(np.median(run) - np.median(largest)) > GAP_SECONDS for run in runs):
        return None
    return float(np.median(largest))


def phase_search(stream, row, expected, length):
    center = round((row["origin_offset"] + expected) * RATE)
    start = center - length // 2
    raw = np.stack([trace.data for trace in stream])
    if start < 0 or start + length > raw.shape[1]:
        return None
    section = raw[:, start:start + length]
    if not np.isfinite(section).all() or any(np.ptp(axis) == 0 for axis in section):
        return None
    # A constant half second around the putative arrival prevents phase confirmation.
    near = section[:, length // 2 - RATE:length // 2 + RATE]
    for axis in near:
        edges = np.r_[0, np.flatnonzero(np.diff(axis)) + 1, len(axis)]
        if np.diff(edges).max() >= RATE // 2:
            return None
    band = stream.copy().filter("bandpass", freqmin=1, freqmax=20)
    filtered = np.stack([trace.data[start:start + length] for trace in band])
    return np.stack([section[::-1], filtered[::-1]]).astype(np.float32), start


def collect(model, device, batch, proposals):
    x = torch.from_numpy(np.concatenate([item[0] for item in batch])).to(device)
    with torch.inference_mode():
        maps = model(model.annotate_batch_pre(x, {})).cpu().numpy().reshape(len(batch), 2, 3, -1)
    for prediction, (_, start, row, phase, expected) in zip(maps, batch, strict=True):
        half = model.in_samples // 2
        width = round(GAP_SECONDS * RATE)
        scores = prediction[:, model.labels.index(phase), half - width:half + width + 1]
        peaks = scores.argmax(1)
        confidence = scores[np.arange(2), peaks]
        arrivals = (start + half - width + peaks) / RATE - row["origin_offset"]
        if (confidence < 0.4).any() or abs(arrivals[0] - arrivals[1]) > 0.3:
            continue
        path = row["paths"][0]
        destination = proposals.setdefault(path, {
            **{name: row[name] for name in ("paths", "group", "station", "origin_offset")},
            "phases": {},
        })
        previous = destination["phases"].get(phase)
        if previous and previous["agrees"]:
            continue
        destination["phases"][phase] = {
            "time": float(arrivals.mean()), "view_times": arrivals.tolist(),
            "view_scores": confidence.tolist(), "agrees": True,
            "array_time": expected, "source": "RED-PAN array and PhaseNet station",
        }


def run(args):
    from seisbench.models import PhaseNet

    split = json.loads(args.split.read_text())
    base = json.loads(args.redpan.read_text())
    split_hash = hashlib.sha256(args.split.read_bytes()).hexdigest()
    if base["split_sha256"] != split_hash:
        raise ValueError("RED-PAN proposals must belong to this training split")
    rows = {row["paths"][0]: row for row in split["events"] if row["split"] == "train"}
    proposals = {row["paths"][0]: row for row in base["records"]}
    if not proposals.keys() <= rows.keys():
        raise ValueError("RED-PAN proposals must come from training waveforms")
    sites = {path: site(row) for path, row in rows.items()}
    seeds = defaultdict(dict)
    for path, row in proposals.items():
        for phase, pick in row["phases"].items():
            if pick["agrees"]:
                key = (row["group"], sites[path], phase)
                seeds[key][row["station"]] = pick["time"]
    times = {key: clustered_time(picks.values()) for key, picks in seeds.items()}
    device = select_device(args.device)
    model = PhaseNet.from_pretrained("stead").to(device).eval()
    batch = []
    for path, row in rows.items():
        for phase in "PS":
            expected = times.get((row["group"], sites[path], phase))
            if expected is None or proposals.get(path, {}).get("phases", {}).get(phase, {}).get("agrees"):
                continue
            stream = load_xyz([ROOT / path for path in row["paths"]])
            if stream is None:
                continue
            candidate = phase_search(stream, row, expected, model.in_samples)
            if candidate is not None:
                batch.append((*candidate, row, phase, expected))
            if len(batch) == 16:
                collect(model, device, batch, proposals)
                batch.clear()
    if batch:
        collect(model, device, batch, proposals)
    counts, added = Counter(), Counter()
    for row in proposals.values():
        picks = row["phases"]
        if all(picks.get(phase, {}).get("agrees") for phase in "PS") and picks["S"]["time"] - picks["P"]["time"] < 2:
            for pick in picks.values():
                if "source" in pick:
                    pick["agrees"] = False
        counts.update(phase for phase, pick in picks.items() if pick["agrees"])
        added.update(phase for phase, pick in picks.items() if pick["agrees"] and "source" in pick)
    result = {**base, "records": list(proposals.values()), "recordings": len(proposals),
              "agreeing": dict(counts), "additional_array_proposals": dict(added),
              "base_sha256": hashlib.sha256(args.redpan.read_bytes()).hexdigest(),
              "station_picker": "PhaseNet STEAD", "station_min_score": 0.4}
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Added {dict(added)} using array and station agreement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "redpan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    run(args)
