"""Plot co-located QSIS instruments on a common origin-relative clock."""

from seismictorch.paths import RESOURCES

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from obspy import UTCDateTime
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz, triplets
from seismictorch.qsis.review import inventory, plt, travel_times

REVIEW_FILTER = butter(4, (0.7, 12), "bandpass", fs=100, output="sos")


def event_arrays(records, limit, min_samples=40000):
    """Use fixed bands for review; do not generate arrival labels."""
    sites = defaultdict(list)
    for row in records:
        stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=min_samples)
        if stream is None:
            continue
        sac = stream[0].stats.sac
        site = (round(float(sac.stla), 3), round(float(sac.stlo), 3))
        origin = row["origin_offset"]
        raw = np.stack([trace.data for trace in stream]).astype(np.float64)
        lo = round((origin - 12) * 100)
        hi = round((origin + limit + 2) * 100)
        if lo < 0 or hi > raw.shape[1] or not np.isfinite(raw[:, lo:hi]).all():
            continue
        segment = raw[:, lo:hi]
        if (np.ptp(segment, axis=1) == 0).any():
            continue
        # Preserve and mark sample repeats before band-pass processing.
        gaps = np.zeros_like(segment, dtype=bool)
        for axis, samples in enumerate(segment):
            edges = np.r_[0, np.flatnonzero(np.diff(samples)) + 1, len(samples)]
            for i in np.flatnonzero(np.diff(edges) >= 100):
                gaps[axis, edges[i]:edges[i + 1]] = True
        if gaps[:, :1000].any():
            continue
        band = sosfiltfilt(REVIEW_FILTER, segment)
        # Compare a station against its own pre-event background.
        scale = np.sqrt(np.mean(band[:, :1000] ** 2, axis=1))
        if np.any(scale <= 1e-8):
            continue
        normalized = band / scale[:, None]
        normalized[gaps] = np.nan
        time = np.arange(len(segment[0])) / 100 + lo / 100 - origin
        target = np.arange(-10 * 100, limit * 100) / 100
        signal = np.stack([np.interp(target, time, axis) for axis in normalized])
        power = uniform_filter1d(np.nan_to_num(signal) ** 2, 35, axis=1)
        envelope = np.sqrt(np.maximum(power, 0))
        envelope[~np.isfinite(signal)] = np.nan
        sites[site].append((row, signal, envelope))
    return sites


def plot_site(path, site, group, traces, limit):
    time = np.arange(-10 * 100, limit * 100) / 100
    signal = np.stack([x[1] for x in traces])
    envelope = np.stack([x[2] for x in traces])
    # Mask dropout-affected cells in all summaries.
    vertical = envelope[:, 2]
    horizontal = np.sqrt((envelope[:, 0] ** 2 + envelope[:, 1] ** 2) / 2)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.6), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": (2, 2, 1.5)})
    for ax, values, title in zip(axes, (vertical, horizontal), ("Vertical energy", "Horizontal energy")):
        values = 20 * np.log10(np.maximum(values, 0.1))
        ax.imshow(values, aspect="auto", origin="lower", interpolation="nearest",
                  extent=(time[0], time[-1], -0.5, len(traces) - 0.5),
                  vmin=-6, vmax=18, cmap="magma")
        ax.set_yticks(range(len(traces)), [x[0]["station"] for x in traces], fontsize=7)
        ax.set_title(title + "  ·  dB relative to each station's pre-event RMS", loc="left", fontsize=10)
    for values, label in ((vertical, "Z"), (horizontal, "XY")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values, axis=0)
        axes[2].plot(time, median, linewidth=0.95, label=label)
    axes[2].set(xlabel="Seconds from the SAC event origin", ylabel="Median RMS ratio")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"{group}  |  {site[0]:.3f}, {site[1]:.3f}  |  {len(signal)} instruments")
    fig.savefig(path, dpi=115)
    plt.close(fig)


def catalog_estimates(taup, row, nearby):
    """Move nearby catalog picks to the QSIS site for display as review context."""
    _, local = travel_times(taup, row)
    _, lat, lon = row["event_key"].split("/")
    lat, lon = float(lat), float(lon)
    estimates = {}
    for phase, picks in nearby.items():
        values = []
        for pick in picks:
            if pick["distance_to_qsis_km"] > 25:
                continue
            degrees = locations2degrees(lat, lon, *pick["station_coords"])
            arrivals = taup.get_travel_times(source_depth_in_km=row["depth_km"],
                                             distance_in_degree=degrees,
                                             phase_list=[phase.lower(), phase, phase + "n"])
            if arrivals:
                values.append(pick["seconds_from_qsis_origin"] + local[phase] - arrivals[0].time)
        estimates[phase] = values
    return estimates, local


def plot_review_detail(path, row, traces, estimates, expected):
    target = next((trace for trace in traces if trace[0]["paths"] == row["paths"]), None)
    if target is None:
        return False
    time = np.arange(target[1].shape[1]) / 100 - 10
    envelope = np.stack([trace[2] for trace in traces])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vertical = np.nanmedian(envelope[:, 2], axis=0)
        horizontal = np.nanmedian(np.sqrt((envelope[:, 0] ** 2 + envelope[:, 1] ** 2) / 2), axis=0)
    fig, axes = plt.subplots(4, 2, figsize=(11, 6.4), sharex="col", layout="constrained")
    for col, phase in enumerate("PS"):
        values = estimates.get(phase, [])
        center = np.median(values) if values else expected[phase]
        visible = (time >= center - 5) & (time <= center + 5)
        for i, label in enumerate("XYZ"):
            ax = axes[i, col]
            ax.plot(time[visible], target[1][i, visible], color="#263b53", linewidth=0.8)
            ax.set_ylabel(f"{label} / noise RMS")
        ax = axes[3, col]
        ax.plot(time[visible], vertical[visible], label="Z", linewidth=1)
        ax.plot(time[visible], horizontal[visible], label="XY", linewidth=1)
        ax.set_ylabel("Array median RMS")
        ax.legend(fontsize=8, loc="upper right")
        ax.set(xlim=(center - 5, center + 5), xlabel="Seconds from SAC event origin")
        for item in axes[:, col]:
            item.grid(alpha=0.2)
            if values:
                item.axvspan(min(values) - 1, max(values) + 1, alpha=0.14, color="#cf9232")
        source = (f"{len(values)} nearby catalog picks, moveout adjusted" if values else
                  row.get("review_context", "TauP search interval"))
        axes[0, col].set_title(f"{phase} review · {source}", fontsize=9)
    fig.suptitle(f"{row['id']} · {row['station']} · {len(traces)} same-site instruments", fontsize=12)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def render_review_details(args, groups, taup, plots, digest):
    manifest = json.loads(args.review_manifest.read_text())
    if not manifest["blind"] or manifest["split"] != args.partition or manifest["split_sha256"] != digest:
        raise ValueError("Review packet is outside this frozen partition")
    frozen = {tuple(row["paths"]): row for records in groups.values() for row in records}
    for row in manifest["records"]:
        source = frozen.get(tuple(row["paths"]))
        if source is None or any(row[key] != source[key] for key in ("group", "event_key", "origin_offset")):
            raise ValueError(f"Unexpected recording in review packet: {row['id']}")
    context = {}
    if args.catalog:
        catalog = json.loads(args.catalog.read_text())
        if catalog["split_sha256"] != digest:
            raise ValueError("ISC context is outside this frozen validation split")
        context = {row["id"]: row.get("nearest", {}) for row in catalog["records"]}
    details, entries = [], []
    for group, records in sorted(groups.items()):
        reviews = [row for row in manifest["records"] if row["group"] == group]
        timing = {row["id"]: catalog_estimates(taup, row, context.get(row["id"], {})) for row in reviews}
        centers = [x for estimates, expected in timing.values()
                   for x in list(expected.values()) + [t for values in estimates.values() for t in values]]
        if not centers:
            continue
        limit = int(np.clip(np.ceil(max(centers) + 8), 35, 150))
        traces = {tuple(item[0]["paths"]): site_records for site_records in event_arrays(records, limit).values()
                  for item in site_records}
        for row in reviews:
            ident = row["id"]
            estimates, expected = timing[ident]
            rendered = plot_review_detail(plots / f"{ident}.png", row, traces.get(tuple(row["paths"]), []),
                                          estimates, expected)
            entries.append({"id": ident, "rendered": rendered, "isc_context_seconds": estimates})
            if rendered:
                details.append(f'<figure><img loading="lazy" src="plots/{ident}.png" '
                               f'alt="{ident}"><figcaption>{ident}</figcaption></figure>')
    (args.output / "details.json").write_text(json.dumps({
        "purpose": "Waveform review with TauP search intervals and same-site recordings.",
        "split": manifest["split"], "manifest_sha256": hashlib.sha256(args.review_manifest.read_bytes()).hexdigest(),
        "split_sha256": digest, "records": entries,
    }, indent=2) + "\n")
    if all(entry["rendered"] for entry in entries):
        payload = json.dumps({**manifest, "catalog_context": True} if args.catalog else manifest, allow_nan=False)
        template = (RESOURCES / "qsis_review.html").read_text(encoding="utf-8")
        (args.output / "annotate.html").write_text(
            template.replace("/*MANIFEST*/null", payload.replace("<", "\\u003c")), encoding="utf-8")
        details.insert(0, '<p><a href="annotate.html">開啟到時標註表單</a></p>')
    return details


def station_packet(source, output, consensus_path, public, allow_single_phase=False):
    """Use array picks to place search intervals for separate station review."""
    array_path = consensus_path.parent / "manifest.json"
    consensus = json.loads(consensus_path.read_text())
    source_path = source / "download.json" if public else source
    hash_key = "download_sha256" if public else "split_sha256"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    expected_split = "public_audit" if public else "train"
    if (consensus["split"] != expected_split or consensus["review_unit"] != "event_array"
            or consensus["manifest_sha256"] != hashlib.sha256(array_path.read_bytes()).hexdigest()
            or consensus[hash_key] != source_hash):
        raise ValueError("Array consensus does not match the source packet")
    if public:
        download = json.loads(source_path.read_text())
        min_samples = 100 * download["selection"].get("min_duration_seconds", 400)
        records = inventory(triplets(source.resolve() / "sac"), min_pga=0, min_samples=min_samples)
    else:
        min_samples = 40000
        records = [row for row in json.loads(source.read_text())["events"] if row["split"] == "train"]
    by_path = {tuple(row["paths"]): row for row in records}
    selected = []
    plots = output / "plots"
    plots.mkdir(parents=True)
    require_pair = public and not allow_single_phase
    taup = None if require_pair else TauPyModel("ak135")
    for event in consensus["records"]:
        agreed = {phase.upper(): pick["time"] for phase, pick in event["phases"].items()
                  if pick["status"] == "agreed"}
        if len(agreed) < (2 if require_pair else 1):
            continue
        records = [by_path[tuple(paths)] for paths in event["recordings"]]
        group_key = "event_key" if public else "group"
        if any(row[group_key] != event["event"] for row in records):
            raise ValueError("Station recording is outside its reviewed event")
        centers = agreed if require_pair else travel_times(taup, records[0])[1] | agreed
        limit = int(np.ceil(max(centers.values()) + 8))
        for traces in event_arrays(records, limit, min_samples).values():
            for row, _, _ in traces:
                ident = f"STATION-{len(selected) + 1:03}"
                record = {**row, "id": ident, "array_id": event["id"],
                          "review_context": "Array or TauP search interval; no pick marker"}
                if plot_review_detail(plots / f"{ident}.png", record, traces, {}, centers):
                    selected.append(record)
    if not selected:
        raise ValueError("No station recordings with sufficient agreed event-array phases")
    manifest = {"blind": True, "split": "public_station_audit" if public else "train",
                "label_context": ("Unmarked station plots centered on earlier array estimates" if require_pair
                                  else "Unmarked station plots using array or TauP search intervals"),
                hash_key: source_hash,
                "array_consensus_sha256": hashlib.sha256(consensus_path.read_bytes()).hexdigest(),
                "seed": 42, "records": selected}
    payload = json.dumps(manifest, indent=2, allow_nan=False)
    (output / "manifest.json").write_text(payload + "\n", encoding="utf-8")
    template = (RESOURCES / "qsis_review.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("/*MANIFEST*/null", payload.replace("<", "\\u003c")),
                                       encoding="utf-8")
    print(f"{len(selected)} station records from {len({r['array_id'] for r in selected})} events")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", type=Path)
    source.add_argument("--public-audit", type=Path, help="Public QSIS download directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--array-consensus", type=Path, help="Make a station review packet from array consensus")
    parser.add_argument("--allow-single-phase", action="store_true",
                        help="Include events with one agreed phase and use TauP for the other search interval")
    parser.add_argument("--training-groups", type=Path,
                        help="JSON list of frozen training groups for an annotation packet")
    args = parser.parse_args()
    if args.output.exists() or (args.catalog and not args.review_manifest):
        parser.error("Use a new output directory and include a review manifest with catalog context")
    if args.catalog and (not args.split or args.partition != "validation"):
        parser.error("The catalog context is only available for the frozen validation packet")
    if args.review_manifest and not args.split:
        parser.error("A frozen review manifest requires --split")
    if args.training_groups and (not args.split or args.review_manifest):
        parser.error("Training groups require --split and a separate packet")
    if args.allow_single_phase and not args.array_consensus:
        parser.error("--allow-single-phase requires --array-consensus")
    if args.array_consensus:
        if args.catalog or args.training_groups:
            parser.error("Use only the source archive and array consensus for the station packet")
        station_packet(args.public_audit or args.split, args.output, args.array_consensus, bool(args.public_audit),
                       args.allow_single_phase)
        return
    audit = args.public_audit is not None
    source_file = args.public_audit / "download.json" if audit else args.split
    digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
    data = json.loads(source_file.read_text())
    min_samples = 40000
    if audit:
        min_samples = 100 * data["selection"].get("min_duration_seconds", 400)
        rows = inventory(triplets(args.public_audit.resolve() / "sac"), min_pga=0, min_samples=min_samples)
    elif args.training_groups:
        requested = set(json.loads(args.training_groups.read_text()))
        rows = [row for row in data["events"] if row["split"] == "train" and row["group"] in requested]
        if not requested or {row["group"] for row in rows} != requested:
            parser.error("Use a nonempty list of frozen training groups")
    else:
        rows = [row for row in data["events"] if row["split"] == args.partition]
    groups = defaultdict(list)
    for row in rows:
        groups[row["event_key"] if audit else row["group"]].append(row)
    plots = args.output / "plots"
    plots.mkdir(parents=True)
    taup = TauPyModel("ak135")
    images = []
    for group, rows in sorted(groups.items()):
        _, predicted = travel_times(taup, rows[0])
        limit = int(np.clip(np.ceil(predicted["S"] + 18), 35, 150))
        for site, traces in sorted(event_arrays(rows, limit, min_samples).items()):
            if len(traces) < 2:
                continue
            ident = f"{'PUBLIC' if audit else 'ARRAY'}-{len(images) + 1:03}"
            traces.sort(key=lambda x: x[0]["station"])
            plot_site(plots / f"{ident}.png", site, group, traces, limit)
            images.append({"id": ident, "event": group, "site": site,
                           "stations": [x[0]["station"] for x in traces],
                           "recordings": [x[0]["paths"] for x in traces]})
    payload = {"purpose": "Marker-free array views. No phase labels.",
               "blind": True, "split": "train" if args.training_groups else args.partition,
               "split_sha256": digest,
               "images": images}
    if audit:
        payload = {"purpose": "Later-period annotation audit; not a benchmark or test set.",
                   "blind": True, "split": "public_audit", "download_sha256": digest,
                   "images": images}
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    cards = "\n".join(f'<figure><img loading="lazy" src="plots/{x["id"]}.png" alt="{x["id"]}">'
                      f'<figcaption>{x["id"]} · {UTCDateTime(x["event"].split("/")[0])}</figcaption></figure>'
                      for x in images)
    if args.review_manifest:
        detail_cards = render_review_details(args, groups, taup, plots, digest)
        guide = ('橘色區域根據鄰站目錄到時和 TauP 走時差估算。' if args.catalog
                 else 'P、S 面板是依 TauP 選出的搜尋區間，請從各台儀器的波形確認初動。')
        cards = ('<h2>逐筆到時複核輔助</h2><p>' + guide
                 + '圖中波形為 0.7–12 Hz 濾波，依震前噪音正規化。</p>'
                 + "\n".join(detail_cards) + '<h2>同址儀器總覽</h2>' + cards)
    title = "QSIS 公開錄音視讀" if audit else f"QSIS {payload['split']} 同址波形"
    (args.output / "index.html").write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>QSIS 同址波形</title>'
        '<style>body{font:16px system-ui;max-width:70rem;margin:2rem auto}img{width:100%}'
        f'figure{{margin:2rem 0}}</style><h1>{title}</h1>'
        '<p>各儀器的振幅以震前背景正規化。圖上未疊加模型候選到時。</p>'
        + cards, encoding="utf-8")
    print(f"{len(images)} same-site plots from {len({x['event'] for x in images})}/{len(groups)} events")


if __name__ == "__main__":
    main()
