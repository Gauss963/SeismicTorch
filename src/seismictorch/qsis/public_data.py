"""Load independently reviewed public QSIS windows for training or diagnostics."""

from collections import Counter
import hashlib
import json
import math

import numpy as np
from obspy import UTCDateTime

from seismictorch.training.pilot import filtered_xyz, window
from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.data.archives import CLASS_NAMES

PARTITION = "public_station_audit"
SHIFTS = (0, -20, 20)  # offsets in the 100 Hz SAC recording


def public_samples(split, download_path, manifest_path, labels_path, shifts=SHIFTS):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    download, manifest, labels = [json.loads(p.read_text()) for p in (download_path, manifest_path, labels_path)]
    if (manifest["split"] != PARTITION or labels["split"] != PARTITION or not manifest["blind"]
            or labels["review_unit"] != "recording" or labels["manifest_sha256"] != sha(manifest_path)
            or manifest["download_sha256"] != sha(download_path)
            or labels["download_sha256"] != manifest["download_sha256"]):
        raise ValueError("The public download, packet and reviews do not match")

    downloaded = {tuple(r["paths"]): r for r in download["records"] if r["status"] == "downloaded"}
    records = {r["id"]: r for r in manifest["records"]}
    reviewed = {r["id"]: r for r in labels["records"]}
    if (records.keys() != reviewed.keys() or len(records) != len(manifest["records"])
            or len(reviewed) != len(labels["records"])):
        raise ValueError("Each station in the audit must have exactly one consensus entry")
    latest_frozen = max(
        max(UTCDateTime(row["origin_utc"]) for row in split["events"]),
        max(UTCDateTime(row["group"]) + 86400 for row in split["noise"]),
    )
    coverage, waveform_hashes, samples = Counter(), {}, []
    min_samples = 100 * download["selection"].get("min_duration_seconds", 400)
    for ident, row in records.items():
        review = reviewed[ident]
        if any(review.get(key) != value for key, value in row.items()):
            raise ValueError(f"Review provenance differs for {ident}")
        paths = tuple(row["paths"])
        origin = UTCDateTime(row["origin_utc"])
        source = downloaded.get(paths)
        if (source is None or source["station"] != row["station"] or origin <= latest_frozen
                or source["event"] != origin.strftime("%Y%m%d%H%M%S")
                or not math.isclose(-source["start_seconds"], row["origin_offset"], abs_tol=0.001)):
            raise ValueError(f"Unexpected public waveform for {ident}")
        picks = [(i, review["phases"][phase]) for i, phase in enumerate("ps")]
        coverage.update(f"{CLASS_NAMES[i]}:{pick['status']}" for i, pick in picks)
        agreed = [(i, pick["time"]) for i, pick in picks if pick["status"] == "agreed"]
        if not agreed:
            continue
        stream = load_xyz([ROOT / path for path in paths], min_samples=min_samples)
        if stream is None:
            raise ValueError(f"Unreadable reviewed SAC waveform: {ident}")
        current_hashes = [sha(ROOT / path) for path in paths]
        if "sha256" in source and source["sha256"] != current_hashes:
            raise ValueError(f"Public SAC contents changed for {ident}")
        waveform_hashes.update(zip(paths, current_hashes, strict=True))
        raw = np.stack([tr.data for tr in stream])
        filtered = filtered_xyz(stream)
        for i, time in agreed:
            if not math.isfinite(time) or not 0 <= time < raw.shape[1] / 100 - row["origin_offset"]:
                raise ValueError(f"Invalid arrival time for {ident}/{CLASS_NAMES[i]}")
            center = round((row["origin_offset"] + time) * 100)
            # Use the same reviewed clips across all three sensitivity conditions.
            clips = [window(raw, filtered, center + shift) for shift in shifts]
            if any(clip is None for clip in clips):
                coverage[f"{CLASS_NAMES[i]}:qc_excluded"] += 1
                continue
            samples.append((clips, {"id": ident, "event_key": row["event_key"], "group": row["event_key"],
                                    "origin_utc": row["origin_utc"], "paths": row["paths"],
                                    "station": row["station"], "label": i, "onset_center": time,
                                    "center_sample": center}))
    return samples, coverage, waveform_hashes, records
