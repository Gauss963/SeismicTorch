"""Prepare a stratified QSIS SAC sample for P/S arrival review."""

from seismictorch.paths import RESOURCES

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from obspy import read
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel

from seismictorch.paths import ROOT
from seismictorch.data.sac import triplets, load_xyz
PHASES = ("P", "S")
COLORS = {"P": "#1760a5", "S": "#b54b12"}
LEGACY_SHIFT = 0.8


def inventory(paths_list, min_pga=2.0, min_samples=40000):
    """Filter usable SAC metadata and compute a pre-origin noise proxy."""
    records = []
    for paths in paths_list:
        stream = load_xyz(paths, min_samples=min_samples)
        if stream is None:
            continue
        tr = stream[0]
        sac = tr.stats.sac
        keys = ("b", "o", "evla", "evlo", "evdp", "stla", "stlo")
        if any(key not in sac or not np.isfinite(sac[key]) or sac[key] == -12345 for key in keys):
            continue
        if (not -90 <= sac.evla <= 90 or not -90 <= sac.stla <= 90
                or not -180 <= sac.evlo <= 180 or not -180 <= sac.stlo <= 180
                or not 0 <= sac.evdp < 700):
            continue
        origin_offset = float(sac.o) - float(sac.b)
        origin = tr.stats.starttime + origin_offset
        if not 12 <= origin_offset <= tr.stats.npts / 100 - 30:
            continue
        xyz = np.stack([axis.data.astype(np.float64) for axis in stream])
        if not np.isfinite(xyz).all():
            continue
        pga = np.linalg.norm(xyz, axis=0).max()
        if not pga > 0 or pga < min_pga:
            continue
        background = xyz[:, int((origin_offset - 12) * 100):int((origin_offset - 2) * 100)]
        centered = background - background.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(centered ** 2, axis=0)))
        event_key = f"{origin.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z/{sac.evla:.3f}/{sac.evlo:.3f}"
        distance = gps2dist_azimuth(sac.stla, sac.stlo, sac.evla, sac.evlo)[0] / 1000
        records.append({
            "paths": [str(path.relative_to(ROOT)) for path in paths],
            "station": paths[0].parent.name, "event_key": event_key,
            "origin_utc": str(origin), "origin_offset": origin_offset,
            "sac_begin": float(sac.b), "distance_km": distance,
            "depth_km": float(sac.evdp), "pga_gal": float(pga),
            "pga_to_background_db": float(20 * np.log10(pga / max(rms, 1e-8))),
        })
    return records


def choose_records(records, count, seed):
    """Cycle through SNR proxy bands; prefer new events and less-used stations."""
    rng = np.random.default_rng(seed)
    ordered = sorted(records, key=lambda row: row["pga_to_background_db"])
    bands = [list(rng.permutation(band)) for band in np.array_split(np.asarray(ordered, dtype=object), 3)]
    quotas = [count // 3 + (i < count % 3) for i in range(3)]
    order = sorted(range(3), key=lambda i: len({row["event_key"] for row in bands[i]}))
    used_events, station_counts, selected = set(), Counter(), []
    for new_event_only in (True, False):
        while len(selected) < count:
            start = len(selected)
            for i in order:
                if quotas[i] == 0:
                    continue
                band = bands[i]
                options = [i for i, row in enumerate(band)
                           if not new_event_only or row["event_key"] not in used_events]
                if options:
                    index = min(options, key=lambda i: station_counts[band[i]["station"]])
                    row = band.pop(index)
                    row["snr_band"] = ("low", "middle", "high")[i]
                    quotas[i] -= 1
                    selected.append(row)
                    used_events.add(row["event_key"])
                    station_counts[row["station"]] += 1
                if len(selected) == count:
                    break
            if len(selected) == start:
                break
    return selected


def travel_times(model, row):
    distance = kilometers2degrees(row["distance_km"])
    args = dict(source_depth_in_km=row["depth_km"], distance_in_degree=distance)
    arrivals = model.get_travel_times(**args, phase_list=["p", "P", "Pn", "s", "S", "Sn"])
    legacy, predicted = {}, {}
    for phase, velocity in zip(PHASES, (6, 3.5)):
        default = (row["distance_km"] + row["depth_km"]) / velocity
        matches = [a.time for a in arrivals if a.name in (phase, phase + "n")]
        legacy[phase] = matches[-1] if matches else default
        predicted[phase] = next((a.time for a in arrivals if a.name.upper().startswith(phase)), default)
    return legacy, predicted


def legacy_windows(stream, row, times):
    """Recreate peak-centered windows before the notebook's random shift."""
    filtered = stream.copy().detrend("linear").taper(0.05).filter("highpass", freq=0.5)
    composite = np.linalg.norm(np.stack([tr.data for tr in filtered]), axis=0)
    ref = int(-row["sac_begin"] * 100)
    p = ref + int(times["P"] * 100)
    s = ref + int(times["S"] * 100) + 1200
    windows = {}
    for phase, start, end in (("P", p, min(p + 1200, s - 100)), ("S", s, s + 1200)):
        # Report an invalid search explicitly; do not use Python negative indexing.
        if not 0 <= start < min(end, len(composite)):
            windows[phase] = None
            continue
        peak = start + int(np.argmax(composite[start:end]))
        center = peak / 100 - row["origin_offset"]
        windows[phase] = [center - 1, center + 1]
    return filtered, windows


def model_picks(model, stream, origin, predicted):
    # Map nominal QSIS axes to PhaseNet's Z/N/E order. Many SAC files do not
    # specify azimuth, so these predictions are candidates for human review.
    renamed = stream.copy()
    for trace, axis in zip(renamed, "ENZ"):
        trace.stats.channel = "HH" + axis
    renamed.trim(origin - 12, origin + max(predicted.values()) + 30)
    output = model.classify(renamed, strict=True, p_threshold=0.3, s_threshold=0.3)
    candidates = {}
    for phase in PHASES:
        nearby = [p for p in output.picks if p.phase == phase
                  and abs((p.peak_time - origin) - predicted[phase]) <= 6]
        pick = max(nearby, key=lambda p: p.peak_value, default=None)
        candidates[phase] = (None if pick is None else
                             {"time": float(pick.peak_time - origin), "score": float(pick.peak_value)})
    return candidates


def render_plot(path, stream, row):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 5), layout="constrained")
    extent = list(row["taup"].values()) + [
        window[1] + LEGACY_SHIFT for window in row["legacy"].values() if window
    ] + [pick["time"] for pick in row["phasenet"].values() if pick]
    blind = row.get("blind", False)
    left, right = min(-3, min(extent) - 5), max(extent) + (12 if blind else 4)
    time = stream[0].times() - row["origin_offset"]
    visible = (time >= left) & (time <= right)
    for axis, trace, component in zip(axes, stream, "XYZ"):
        axis.plot(time[visible], trace.data[visible], color="#252c34", linewidth=0.62)
        axis.set_ylabel(f"{component} (gal)")
        axis.grid(axis="x", alpha=0.18)
        for phase in (() if blind else PHASES):
            color = COLORS[phase]
            window = row["legacy"][phase]
            if window:
                lo, hi = window
                axis.axvspan(lo - LEGACY_SHIFT, hi + LEGACY_SHIFT, color=color, alpha=0.08)
                axis.axvspan(lo, hi, color=color, alpha=0.19)
            axis.axvline(row["taup"][phase], color=color, linestyle=":", linewidth=1.1)
            if row["phasenet"][phase]:
                axis.axvline(row["phasenet"][phase]["time"], color=color, linestyle="--", linewidth=1.2)
    axes[-1].set(xlim=(left, right), xlabel="Seconds from SAC event origin")
    handles = [Patch(color=COLORS[p], alpha=0.3, label=f"Legacy {p} window") for p in PHASES]
    handles.extend([Patch(color="#64748b", alpha=0.12, label="Possible jitter envelope"),
                    Line2D([], [], color="#252c34", ls=":", label="TauP"),
                    Line2D([], [], color="#252c34", ls="--", label="PhaseNet")])
    if not blind:
        axes[0].legend(handles=handles, ncol=5, loc="upper right", fontsize=8)
    fig.suptitle(f"{row['id']}  |  {row['station']}  |  {row['origin_utc']}  |  "
                 f"PGA {row['pga_gal']:.1f} gal", fontsize=12)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taup-only", action="store_true", help="Skip PhaseNet inference")
    args = parser.parse_args()
    if args.count < 1 or args.output.exists():
        parser.error("Specify a positive count and a new output directory")

    event_paths = triplets(ROOT / "data_QSIS_Event")
    noise_paths = triplets(ROOT / "data_QSIS_Noise")
    eligible = inventory(event_paths)
    if not eligible:
        raise ValueError("No aligned event recordings with valid SAC metadata and PGA >= 2 gal")
    selected = choose_records(eligible, args.count, args.seed)
    if len(selected) != args.count:
        raise ValueError(f"Can only select {len(selected)} recordings across the requested SNR bands")
    print(f"{len(event_paths)} event triplets, {len(noise_paths)} noise triplets, "
          f"{len(eligible)} eligible, {len(selected)} selected", flush=True)
    picker = None
    if not args.taup_only:
        import torch
        from seisbench.models import PhaseNet

        torch.set_num_threads(2)
        picker = PhaseNet.from_pretrained("stead").eval()
    taup = TauPyModel("ak135")
    plots = args.output / "plots"
    plots.mkdir(parents=True)
    for index, row in enumerate(selected, 1):
        row["id"] = f"QSIS-{index:03d}"
        stream = load_xyz([ROOT / path for path in row["paths"]])
        origin = stream[0].stats.starttime + row["origin_offset"]
        legacy, row["taup"] = travel_times(taup, row)
        filtered, row["legacy"] = legacy_windows(stream, row, legacy)
        row["phasenet"] = (model_picks(picker, stream, origin, row["taup"]) if picker is not None
                          else dict.fromkeys(PHASES))
        render_plot(plots / f"{row['id']}.png", filtered, row)
        if index % 10 == 0 or index == len(selected):
            print(f"Rendered {index}/{len(selected)}", flush=True)
    manifest = {
        "seed": args.seed, "event_triplets": len(event_paths), "noise_triplets": len(noise_paths),
        "eligible": len(eligible), "phasenet_weights": None if picker is None else "stead",
        "records": selected,
    }
    payload = json.dumps(manifest, indent=2, allow_nan=False)
    (args.output / "manifest.json").write_text(payload + "\n", encoding="utf-8")
    template = (RESOURCES / "qsis_review.html").read_text(encoding="utf-8")
    (args.output / "index.html").write_text(template.replace("/*MANIFEST*/null", payload.replace("<", "\\u003c")),
                                            encoding="utf-8")
    print(f"Open {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
