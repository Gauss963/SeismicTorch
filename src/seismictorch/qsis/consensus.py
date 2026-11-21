"""Merge independent blind waveform reviews without using model predictions."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def consensus(manifest, reviews, tolerance=0.6):
    packet = manifest["images"] if "images" in manifest else manifest["records"]
    source = {row["id"]: row for row in packet}
    per_record = {ident: [] for ident in source}
    for reviewer in reviews:
        seen = set()
        for row in reviewer:
            ident = row["id"]
            if ident not in source or ident in seen:
                raise ValueError(f"Unknown or repeated review ID: {ident}")
            seen.add(ident)
            for phase in "ps":
                value = row[phase]
                level = row[phase + "_confidence"]
                if level not in {"clear", "tentative", "unclear"} or (
                    value is None and level != "unclear"
                ) or (value is not None and (level == "unclear" or not math.isfinite(value))):
                    raise ValueError(f"Invalid pick for {ident}/{phase}")
            per_record[ident].append(row)
    records = []
    counts = {phase: Counter() for phase in "ps"}
    for ident, row in source.items():
        readings = per_record[ident]
        if len(readings) < 2:
            raise ValueError(f"{ident} needs two independent reviews")
        phases = {}
        for phase in "ps":
            clear = [r[phase] for r in readings if r[phase + "_confidence"] == "clear"]
            if len(clear) < 2:
                status, time = "insufficient_clear", None
            elif max(clear) - min(clear) > tolerance:
                status, time = "disagreement", None
            else:
                status, time = "agreed", median(clear)
            phases[phase] = {"status": status, "time": time, "clear_times": clear}
        p, s = (phases[phase]["time"] for phase in "ps")
        if p is not None and s is not None and s - p < 2:
            for phase in "ps":
                phases[phase].update(status="overlapping", time=None)
        for phase in "ps":
            counts[phase][phases[phase]["status"]] += 1
        records.append({**row, "phases": phases, "reviews": readings})
    return records, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--catalog-context", type=Path,
                        help="Declare that reviewers saw catalog-assisted waveform plots")
    args = parser.parse_args()
    if len({path.resolve() for path in args.reviews}) != len(args.reviews):
        parser.error("Each input must be a different review")
    manifest = json.loads(args.manifest.read_text())
    if not manifest["blind"]:
        parser.error("The reference packet must suppress candidate arrivals")
    context_hash = None
    if args.catalog_context:
        context = json.loads(args.catalog_context.read_text())
        if (manifest["split"] != "validation" or context["split_sha256"] != manifest["split_sha256"]
                or {r["id"] for r in context["records"] if r["rendered"]}
                != {r["id"] for r in manifest["records"]}):
            parser.error("Catalog context must cover exactly the frozen validation packet")
        context_hash = hashlib.sha256(args.catalog_context.read_bytes()).hexdigest()
    records, counts = consensus(manifest, [json.loads(path.read_text()) for path in args.reviews])
    artifact = {
        "label_source": ("AI visual review with shared nearby-station catalog context" if context_hash
                         else manifest.get("label_context", "Independent blind AI visual review")),
        "catalog_context_sha256": context_hash,
        "split": manifest["split"],
        "review_unit": "event_array" if "images" in manifest else "recording",
        "split_sha256": manifest.get("split_sha256"),
        "download_sha256": manifest.get("download_sha256"),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "review_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.reviews],
        "tolerance_seconds": 0.6,
        "counts": counts,
        "records": records,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
