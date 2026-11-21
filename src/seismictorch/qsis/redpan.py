"""Generate weak RED-PAN proposals for training or a separate public audit."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from obspy.taup import TauPyModel

from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.qsis.review import PHASES, inventory, travel_times

RATE = 100
WINDOW = 60 * RATE
MIN_SCORE = 0.5
MAX_VIEW_GAP = 0.35


def records(rows, min_samples=40000):
    taup = TauPyModel("ak135")
    for row in rows:
        stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=max(min_samples, WINDOW))
        if stream is None:
            continue
        raw = np.stack([trace.data for trace in stream], axis=1)
        if not np.isfinite(raw).all() or (np.ptp(raw, axis=0) == 0).any():
            continue
        filtered = stream.copy().filter("bandpass", freqmin=1, freqmax=45)
        band = np.stack([trace.data for trace in filtered], axis=1)
        _, expected = travel_times(taup, row)
        for index, phase in enumerate(PHASES):
            sample = round((expected[phase] + row["origin_offset"]) * RATE)
            start = max(0, min(sample - WINDOW // 2, len(raw) - WINDOW))
            # Compare raw and band-pass predictions using identical waveform support.
            clip = np.stack((raw[start:start + WINDOW], band[start:start + WINDOW])).astype(np.float32)
            deviation = clip.std(axis=1, keepdims=True)
            if (deviation == 0).any():
                continue
            clip = (clip - clip.mean(axis=1, keepdims=True)) / deviation
            yield clip, (row, phase, index, sample, start)


def collect(model, entries, proposals):
    clips = np.concatenate([clip for clip, _ in entries])
    phases, _ = model([clips], training=False)
    phases = np.asarray(phases).reshape(len(entries), 2, WINDOW, 3)
    for prediction, (_, (row, phase, index, sample, start)) in zip(phases, entries):
        low = max(0, sample - start - 8 * RATE)
        high = min(WINDOW, sample - start + 8 * RATE + 1)
        if low >= high:
            continue
        score_map = prediction[:, low:high, index]
        peaks = score_map.argmax(axis=1)
        score = score_map[np.arange(2), peaks]
        time = (start + low + peaks) / RATE - row["origin_offset"]
        key = row["paths"][0]
        if key not in proposals:
            proposals[key] = {name: row[name] for name in ("paths", "group", "station", "origin_offset")}
            proposals[key]["phases"] = {}
        accepted = bool((score >= MIN_SCORE).all() and abs(time[0] - time[1]) <= MAX_VIEW_GAP)
        proposals[key]["phases"][phase] = {
            "time": float(time.mean()), "view_times": time.tolist(), "view_scores": score.tolist(),
            "agrees": accepted,
        }


def public_records(download):
    if download.get("source") != "https://smartsensors.earth.sinica.edu.tw/Events":
        raise ValueError("Use a download manifest from the public QSIS event portal")
    rows = [row for row in download["records"] if row["status"] == "downloaded"]
    paths = [[(ROOT / path).resolve() for path in row["paths"]] for row in rows]
    if any(not path.is_relative_to(ROOT / "runs") for triplet in paths for path in triplet):
        raise ValueError("Public audit waveforms must be under the runs directory")
    records_by_path = {tuple(row["paths"]): row for row in inventory(
        paths, min_pga=0, min_samples=100 * download["selection"].get("min_duration_seconds", 400)
    )}
    for row in rows:
        record = records_by_path.get(tuple(row["paths"]))
        if record is None:
            # Some older recordings start after the origin or have no pre-event background.
            continue
        if record["station"] != row["station"]:
            raise ValueError("Public station metadata does not match the download")
        yield {**record, "group": record["event_key"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", type=Path)
    source.add_argument("--public-download", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = tf.keras.models.load_model(args.weights, compile=False)
    path = args.split or args.public_download
    data = json.loads(path.read_text())
    if args.split:
        rows = [row for row in data["events"] if row["split"] == "train"]
        provenance = {"split_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        min_samples = 40000
    else:
        rows = list(public_records(data))
        min_samples = 100 * data["selection"].get("min_duration_seconds", 400)
        provenance = {"download_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "purpose": "public_review_candidates_only"}
    proposals, batch = {}, []
    for entry in records(rows, min_samples):
        batch.append(entry)
        if len(batch) == 16:
            collect(model, batch, proposals)
            batch.clear()
            if len(proposals) % 80 == 0:
                print(f"Processed {len(proposals)} recordings", flush=True)
    if batch:
        collect(model, batch, proposals)

    counts = Counter()
    for row in proposals.values():
        p, s = (row["phases"].get(phase) for phase in PHASES)
        if p and s and p["agrees"] and s["agrees"] and s["time"] - p["time"] < 2:
            p["agrees"] = s["agrees"] = False
        counts.update(phase for phase, pick in row["phases"].items() if pick["agrees"])
    result = {
        **provenance,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "recordings": len(proposals), "agreeing": dict(counts),
        "min_score": MIN_SCORE, "max_view_gap_seconds": MAX_VIEW_GAP,
        "records": list(proposals.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print({key: value for key, value in result.items() if key != "records"})


if __name__ == "__main__":
    main()
