"""Freeze QSIS evaluation groups before preparing phase labels."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read

from seismictorch.paths import ROOT
from seismictorch.data.sac import triplets
from seismictorch.qsis.review import inventory


def group_events(records):
    """Keep nearby SAC origin times together, even across station IDs."""
    previous = None
    group = None
    for row in sorted(records, key=lambda r: UTCDateTime(r["origin_utc"]).ns):
        origin = UTCDateTime(row["origin_utc"])
        # Conservatively merge nearby origins rather than risk event leakage.
        if previous is None or origin - previous > 5:
            group = str(origin)
        row["group"] = group
        previous = origin
    return records


def allocate_groups(group_ids, seen, validation_count, test_count, rng):
    new = rng.permutation(sorted(set(group_ids) - seen))
    if len(new) < validation_count + test_count:
        raise ValueError("Too few unseen groups for the requested split")
    result = dict.fromkeys(group_ids, "train")
    result.update(dict.fromkeys(new[:test_count], "test"))
    result.update(dict.fromkeys(new[test_count:test_count + validation_count], "validation"))
    return result


def event_splits(records, seen_event_keys, rng):
    groups = defaultdict(list)
    for row in group_events(records):
        groups[row["group"]].append(row)
    seen = {group for group, rows in groups.items()
            if any(row["event_key"] in seen_event_keys for row in rows)}
    allocations = dict.fromkeys(seen, "train")
    unseen = set(groups) - seen
    # Preserve the historical PGA regime, with weaker events as a separate stratum.
    above_gate = {key for key in unseen if max(row["pga_gal"] for row in groups[key]) >= 2}
    for stratum in (above_gate, unseen - above_gate):
        ordered = sorted(stratum, key=lambda key: (
            max(row["pga_to_background_db"] for row in groups[key]), key,
        ))
        for band in np.array_split(ordered, 3):
            count = round(len(band) * 0.2)
            allocations.update(allocate_groups(band, set(), count, count, rng))
    for row in records:
        row["split"] = allocations[row["group"]]
    return records


def noise_splits(paths_list, seen_days, rng):
    records = []
    for paths in paths_list:
        trace = read(str(paths[0]), format="SAC", headonly=True)[0]
        day = str(trace.stats.starttime.date)
        # Exclude recordings that bridge two date groups.
        if str(trace.stats.endtime.date) != day:
            continue
        records.append({"paths": [str(path.relative_to(ROOT)) for path in paths], "group": day})
    groups = {row["group"] for row in records}
    remaining = len(groups - seen_days)
    if remaining < 2:
        raise ValueError("Need at least two previously unused noise dates")
    # Balance unseen dates by recording count without mixing a date across splits.
    counts = Counter(row["group"] for row in records)
    new = rng.permutation(sorted(groups - seen_days)).tolist()
    total = sum(counts[day] for day in new)
    test_days = min(combinations(new, (remaining + 1) // 2),
                    key=lambda days: abs(total - 2 * sum(counts[day] for day in days)))
    allocations = dict.fromkeys(groups, "train")
    for day in new:
        allocations[day] = "test" if day in test_days else "validation"
    for row in records:
        row["split"] = allocations[row["group"]]
    return records


def describe(records):
    names = ("train", "validation", "test")
    samples = Counter(row["split"] for row in records)
    groups = Counter(split for split, _ in {(r["split"], r["group"]) for r in records})
    return {name: {"recordings": samples[name], "groups": groups[name]} for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="Prior review manifest")
    parser.add_argument("--pilot", type=Path, required=True, help="Prior pilot metrics")
    parser.add_argument("--output", type=Path, required=True, help="New split manifest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    review = json.loads(args.review.read_text())
    pilot = json.loads(args.pilot.read_text())
    seen_events = {row["event_key"] for row in review["records"]}
    seen_days = {row["group"] for row in pilot["samples"] if row["label"] == 2}
    if args.output.exists():
        parser.error(f"Split manifest already exists: {args.output}")

    events = event_splits(inventory(triplets(ROOT / "data_QSIS_Event"), min_pga=0), seen_events, rng)
    noise = noise_splits(triplets(ROOT / "data_QSIS_Noise"), seen_days, rng)
    counts = {"events": describe(events), "noise": describe(noise)}
    data = {
        "seed": args.seed,
        "eligibility": "Aligned 100 Hz SAC triplets, valid origin, >=400 s, nonzero PGA",
        "counts": counts,
        "events": events,
        "noise": noise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
