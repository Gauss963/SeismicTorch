"""Prepare waveform-only packets for independent QSIS phase annotation."""

from seismictorch.paths import RESOURCES

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from obspy.taup import TauPyModel

from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.qsis.review import PHASES, render_plot, travel_times


def prepare(split_path, output, partition, per_event, proposal_path=None, supplement_path=None):
    split = json.loads(split_path.read_text())
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    candidates = None
    original_paths = set()
    if supplement_path:
        original = json.loads(supplement_path.read_text())
        if not original["blind"] or original["split"] != partition or original["split_sha256"] != digest:
            raise ValueError("Supplement requires an original packet from the same frozen partition")
        original_paths = {tuple(row["paths"]) for row in original["records"]}
    if proposal_path is not None:
        proposals = json.loads(proposal_path.read_text())
        if proposals["split_sha256"] != digest:
            raise ValueError("Proposal split hash does not match")
        candidates = {r["paths"][0]: sum(p["agrees"] for p in r["phases"].values())
                      for r in proposals["records"]}
    groups = defaultdict(list)
    for row in split["events"]:
        if row["split"] == partition and (candidates is None or candidates.get(row["paths"][0], 0)):
            groups[row["group"]].append(row)
    rng = np.random.default_rng(split["seed"])
    plots = output / "plots"
    plots.mkdir(parents=True)
    taup = TauPyModel("ak135")
    selected = []
    skipped = []
    for group, rows in sorted(groups.items()):
        stations = set()
        ordered = list(rng.permutation(rows))
        if candidates is not None:
            ordered.sort(key=lambda row: -candidates[row["paths"][0]])
        if supplement_path:
            ordered = [row for row in ordered if tuple(row["paths"]) not in original_paths]
            ordered.sort(key=lambda row: (-row["pga_to_background_db"], -row["pga_gal"]))
        for row in ordered:
            if row["station"] in stations:
                continue
            stream = load_xyz([ROOT / path for path in row["paths"]])
            if stream is None or any(np.ptp(tr.data) == 0 for tr in stream):
                continue
            _, predicted = travel_times(taup, row)
            prefix = "SUP" if supplement_path else partition.upper()
            ident = f"{prefix}-{len(selected) + 1:03}"
            visible = dict(id=ident, blind=True, taup=predicted,
                           legacy=dict.fromkeys(PHASES), phasenet=dict.fromkeys(PHASES))
            filtered = stream.detrend("linear").taper(0.05).filter("highpass", freq=0.5)
            render_plot(plots / f"{ident}.png", filtered, row | visible)
            selected.append(row | {"id": ident})
            stations.add(row["station"])
            if len(stations) == per_event:
                break
        if not stations:
            skipped.append(group)
    manifest = {
        "blind": True, "split": partition, "seed": split["seed"],
        "split_sha256": digest,
        "events_without_eligible_record": skipped,
        "records": selected,
    }
    if supplement_path:
        manifest["parent_manifest_sha256"] = hashlib.sha256(supplement_path.read_bytes()).hexdigest()
        manifest["selection"] = "Strongest amplitude-to-background proxy among unreviewed recordings per event"
    payload = json.dumps(manifest, indent=2, allow_nan=False)
    (output / "manifest.json").write_text(payload + "\n", encoding="utf-8")
    template = (RESOURCES / "qsis_review.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("/*MANIFEST*/null", payload.replace("<", "\\u003c")),
                                       encoding="utf-8")
    print(f"{partition}: {len(selected)} recordings from {len(groups) - len(skipped)} event groups; "
          f"{len(skipped)} groups without eligible recordings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--proposals", type=Path, help="Shortlist training recordings for review")
    parser.add_argument("--per-event", type=int, default=2)
    parser.add_argument("--supplement", type=Path, help="Prepare additional recordings from the same frozen events")
    args = parser.parse_args()
    if args.output.exists() or args.per_event < 1:
        parser.error("Specify a new output directory and positive recordings per event")
    if (args.proposals is not None) != (args.partition == "train"):
        parser.error("--proposals is only used for the training review shortlist")
    if args.supplement and args.partition == "train":
        parser.error("Supplemental packets are for validation or test reference review")
    prepare(args.split, args.output, args.partition, args.per_event, args.proposals, args.supplement)


if __name__ == "__main__":
    main()
