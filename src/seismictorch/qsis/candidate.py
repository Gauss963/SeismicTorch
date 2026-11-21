"""Fit a development candidate or score a frozen QSIS review export."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from seismictorch.common import digest, save_json
from seismictorch.data.preprocessing import prepared_waveform
from seismictorch.models.waveform import WAVEFORM_MODELS
from seismictorch.data.orientation import validate_calibrations
from seismictorch.training.pilot import WAVEFORM_FILTERS, event_samples, filtered_xyz, fit, noise_samples, trigger_noise, window
from seismictorch.qsis.public_data import public_samples
from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.training.transfer import extra_samples, reviewed_training_samples
from seismictorch.training.runtime import select_device
from seismictorch.data.archives import CLASS_NAMES


def training_triggers(split, pool_path, split_hash, existing_ids):
    pool = json.loads(pool_path.read_text())
    if pool["split_sha256"] != split_hash:
        raise ValueError("The trigger pool belongs to a different QSIS split")
    mapping = {row["paths"][0]: row for row in split["noise"] if row["split"] == "train"}
    used = set(existing_ids)
    samples = []
    for day, names in sorted(pool["hard_noise_training_pool"].items()):
        for name in names:
            row = mapping.get(name)
            ident = Path(name).name
            if row is None or row["group"] != day or ident in used:
                raise ValueError(f"Invalid, held-out, or repeated training trigger: {name}")
            paths = [ROOT / path for path in row["paths"]]
            stream = load_xyz(paths, min_samples=700)
            trigger = trigger_noise(stream, min_ratio=3.01) if stream is not None else None
            if trigger is None or str(stream[0].stats.starttime.date) != day:
                raise ValueError(f"Could not reproduce training trigger: {name}")
            clip, center = trigger
            samples.append((clip, {"label": 2, "group": day, "paths": row["paths"],
                                   "sha256": [digest(path) for path in paths], "center_sample": center}))
            used.add(ident)
    if not samples:
        raise ValueError("The trigger pool has no training examples")
    return samples


def recut_training_clip(row, mode, orientation=None):
    paths = [ROOT / path for path in row["paths"]]
    stream = load_xyz(paths, min_samples=700)
    if stream is None:
        raise ValueError(f"Could not reload training clip: {paths[0]}")
    center = row["center_sample"]
    raw, filtered, _ = prepared_waveform(stream, center, mode, orientation)
    clip = window(raw, filtered, center)
    if clip is None:
        raise ValueError(f"Could not filter the original training center: {paths[0]}")
    return clip


def train_candidate(args):
    split = json.loads(args.split.read_text())
    frozen_hash = digest(args.split)
    packets = {}
    for partition in ("validation", "test"):
        path = getattr(args, partition + "_manifest")
        manifest = json.loads(path.read_text())
        if not manifest["blind"] or manifest["split"] != partition or manifest["split_sha256"] != frozen_hash:
            raise ValueError(f"Invalid frozen {partition} packet")
        packets[partition] = digest(path)
    orientation = None
    if args.gravity_align:
        calibrations = json.loads(args.orientation_calibration.read_text()) if args.orientation_calibration else []
        validate_calibrations(calibrations)
        orientation = {"method": "gravity-v1", "calibrations": calibrations}
    recut = args.waveform_filter != "tapered" or orientation is not None
    event = event_samples(args.review, with_source=recut)
    groups = {row["group"] for _, _, row in event}
    negatives = noise_samples(2 * len(groups), split["seed"], with_source=recut)
    development = [(x, row) for _, x, row in event + negatives]
    partitions = {r["event_key"]: r["split"] for r in split["events"]}
    noise_train = {r["group"] for r in split["noise"] if r["split"] == "train"}
    if any((row["group"] not in noise_train if row["label"] == 2
            else partitions.get(row["group"]) != "train") for _, row in development):
        raise ValueError("Development input overlaps the frozen evaluation groups")
    extra = extra_samples(args.split, args.proposals, groups, embargo=120) if args.proposals else []
    reviewed = (reviewed_training_samples(args.split, *args.station_review, groups, embargo=120)
                if args.station_review else [])
    if args.station_review and not reviewed:
        raise ValueError("No additional reviewed station arrivals passed training QC")
    public = []
    if args.public_review:
        arrivals, _, waveform_hashes, _ = public_samples(split, *args.public_review)
        if not arrivals:
            raise ValueError("No public station arrivals passed training QC")
        public = [(views[0], row) for views, row in arrivals]
    additional_public, extra_public_receipts = [], []
    public_keys = {(tuple(row["paths"]), row["label"], row["center_sample"]) for _, row in public}
    for paths in args.extra_public_review:
        arrivals, _, hashes, _ = public_samples(split, *paths)
        if not arrivals:
            raise ValueError("An additional public review has no usable agreed arrivals")
        for views, row in arrivals:
            key = (tuple(row["paths"]), row["label"], row["center_sample"])
            if key in public_keys:
                raise ValueError("The public training reviews contain a repeated phase window")
            public_keys.add(key)
            additional_public.append((views[0], row))
        extra_public_receipts.append({"sha256": [digest(path) for path in paths], "waveform_sha256": hashes})
    triggers = (training_triggers(split, args.trigger_pool, frozen_hash,
                                  {row["id"] for _, row in development if row["label"] == 2})
                if args.trigger_pool else [])
    inputs, rows = zip(*(development + extra + reviewed + public + triggers + additional_public))
    if recut:
        inputs = [recut_training_clip(row, args.waveform_filter, orientation) for row in rows]
    labels = [row["label"] for row in rows]
    data = TensorDataset(torch.from_numpy(np.stack(inputs)), torch.tensor(labels))
    weights = torch.load(args.init, map_location="cpu", weights_only=True)
    device = select_device(args.device)
    model = fit(data, list(range(len(labels))), weights, device, args, balance=True, augment=args.augment)
    args.output.mkdir(parents=True)
    checkpoint = args.output / "model.pth"
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, checkpoint)
    save_json(args.output / "training.json", {
        "status": "development_candidate_unvalidated", "model": args.model, "seed": args.seed,
        "waveform_filter": args.waveform_filter,
        "orientation": orientation,
        "orientation_calibration_sha256": digest(args.orientation_calibration) if args.orientation_calibration else None,
        "epochs": args.epochs, "device": str(device), "checkpoint_sha256": digest(checkpoint),
        "noise_selection_seed": split["seed"],
        "split_sha256": frozen_hash, "packet_sha256": packets, "init_sha256": digest(args.init),
        "proposals_sha256": digest(args.proposals) if args.proposals else None,
        "station_review_sha256": [digest(path) for path in args.station_review] if args.station_review else None,
        "public_review": ({"sha256": [digest(path) for path in args.public_review],
                           "waveform_sha256": waveform_hashes} if args.public_review else None),
        "extra_public_reviews": extra_public_receipts,
        "trigger_pool_sha256": digest(args.trigger_pool) if triggers else None,
        "training_classes": dict(Counter(labels)), "sources": rows,
        "augmentation": "±0.2 s shift and ±30° horizontal rotation" if args.augment else None,
    })
    print(f"Saved unvalidated development candidate from {len(labels)} training clips")


def reference_samples(split, manifest, export, waveform_filter="tapered", orientation=None):
    partition = manifest["split"]
    lookup = {r["paths"][0]: r for r in split["events"]}
    records = {r["id"]: r for r in manifest["records"]}
    if set(export["reviews"]) != set(records):
        raise ValueError("Review each packet record. Leave unclear phases blank.")
    samples, coverage = [], Counter()
    for ident, row in records.items():
        frozen = lookup.get(row["paths"][0])
        if frozen is None or frozen["split"] != partition or any(
            frozen[key] != row[key] for key in ("paths", "group", "event_key", "origin_offset")
        ):
            raise ValueError(f"Reference outside the frozen {partition} packet: {ident}")
        label = export["reviews"][ident]
        if label["paths"] != row["paths"] or label["event_key"] != row["event_key"]:
            raise ValueError(f"Review provenance mismatch for {ident}")
        p, s = label["p"], label["s"]
        if p is not None and s is not None and p >= s:
            raise ValueError(f"P must precede S for {ident}")
        stream = load_xyz([ROOT / path for path in row["paths"]])
        if stream is None:
            raise ValueError(f"Unavailable reviewed SAC waveform: {ident}")
        raw = np.stack([tr.data for tr in stream])
        filtered = filtered_xyz(stream, waveform_filter)
        for phase, time in enumerate((p, s)):
            if time is None:
                coverage[f"unresolved_{phase}"] += 1
                continue
            if not math.isfinite(time) or not 0 <= time < len(raw[0]) / 100 - row["origin_offset"]:
                raise ValueError(f"Invalid arrival time for {ident}/{CLASS_NAMES[phase]}")
            center = round((row["origin_offset"] + time) * 100)
            if orientation is not None:
                raw, filtered, _ = prepared_waveform(stream, center, waveform_filter, orientation)
            x = window(raw, filtered, center)
            if x is None:
                coverage[f"qc_excluded_{phase}"] += 1
                continue
            samples.append((x, phase, "event:" + row["group"], ident))
    for row in split["noise"]:
        if row["split"] != partition:
            continue
        stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=700)
        if stream is None:
            coverage["qc_excluded_noise"] += 1
            continue
        clip = trigger_noise(stream)
        if clip is None:
            coverage["qc_excluded_noise"] += 1
            continue
        x, center = clip
        if waveform_filter != "tapered" or orientation is not None:
            raw, filtered, _ = prepared_waveform(stream, center, waveform_filter, orientation)
            x = window(raw, filtered, center)
            if x is None:
                coverage["qc_excluded_noise"] += 1
                continue
        samples.append((x, 2, "noise:" + row["group"], row["paths"][0]))
    return samples, dict(coverage)


def grouped_accuracy(truth, predictions, groups, seed=42, replicates=10000):
    """Resample earthquake groups and noise dates within their own strata."""
    correct = np.asarray(truth) == predictions
    by_group = {}
    for group, outcome in zip(groups, correct, strict=True):
        counts = by_group.setdefault(group, np.zeros(2, dtype=int))
        counts += (int(outcome), 1)
    rng = np.random.default_rng(seed)
    samples = np.zeros((replicates, 2), dtype=int)
    group_counts = {}
    for stratum in ("event", "noise"):
        counts = np.array([count for key, count in sorted(by_group.items())
                           if key.startswith(stratum + ":")])
        group_counts[stratum] = len(counts)
        if len(counts) < 2:
            return None
        indices = rng.integers(len(counts), size=(replicates, len(counts)))
        samples += counts[indices].sum(axis=1)
    scores = samples[:, 0] / samples[:, 1]
    return {"confidence": 0.95, "percentile_interval": np.quantile(scores, [0.025, 0.975]).tolist(),
            "groups": group_counts, "replicates": replicates, "seed": seed}


def review_export(labels, manifest, manifest_hash):
    if "reviews" in labels:
        return labels
    if (manifest["split"] != "validation" or labels.get("review_unit", "recording") != "recording"
            or labels.get("manifest_sha256") != manifest_hash):
        raise ValueError("Consensus labels can only be used for their frozen validation packet")
    reviews = {}
    for row in labels["records"]:
        if row["id"] in reviews:
            raise ValueError(f"Repeated consensus record: {row['id']}")
        review = {key: row[key] for key in ("paths", "event_key")}
        for phase in "ps":
            pick = row["phases"][phase]
            review[phase] = pick["time"] if pick["status"] == "agreed" else None
        reviews[row["id"]] = review
    return {**labels, "reviews": reviews}


def score_candidate(args):
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    frozen_hash = digest(args.split)
    split = json.loads(args.split.read_text())
    manifest = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels.read_text())
    training = json.loads((args.candidate / "training.json").read_text())
    if manifest["split"] not in {"validation", "test"} or not manifest["blind"]:
        raise ValueError("Use an independent validation or test packet")
    if any(data["split_sha256"] != frozen_hash for data in (manifest, labels, training)):
        raise ValueError("Evaluation split hash does not match")
    manifest_hash = digest(args.manifest)
    if manifest_hash != training["packet_sha256"][manifest["split"]]:
        raise ValueError("Evaluation manifest differs from the frozen review packet")
    if labels["split"] != manifest["split"]:
        raise ValueError("Review export belongs to a different partition")
    if manifest["split"] == "test" and labels.get("catalog_context"):
        raise ValueError("The catalog-assisted workflow is restricted to validation")
    export = review_export(labels, manifest, manifest_hash)
    checkpoint = args.candidate / "model.pth"
    if digest(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("Checkpoint digest differs from the training record")
    waveform_filter = training.get("waveform_filter", "tapered")
    samples, coverage = reference_samples(split, manifest, export, waveform_filter, training.get("orientation"))
    if {row[1] for row in samples} != {0, 1, 2}:
        raise ValueError("Scoring requires readable P, S and noise references")
    inputs, truth, groups, ids = zip(*samples)
    model = WAVEFORM_MODELS[training["model"]]()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(np.stack(inputs))).argmax(1).numpy()
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (truth, predictions), 1)
    save_json(args.output, {
        "partition": manifest["split"], "label_provenance": args.label_provenance,
        "label_source": labels.get("label_source", "Review packet export"),
        "catalog_assisted": bool(labels.get("catalog_context") or labels.get("catalog_context_sha256")),
        "split_sha256": frozen_hash, "labels_sha256": digest(args.labels),
        "checkpoint_sha256": digest(checkpoint), "classes": CLASS_NAMES,
        "waveform_filter": waveform_filter,
        "orientation": training.get("orientation"),
        "accuracy": float(matrix.trace() / matrix.sum()),
        "grouped_accuracy_interval": grouped_accuracy(truth, predictions, groups),
        "recall": (matrix.diagonal() / matrix.sum(axis=1)).tolist(),
        "confusion_matrix": matrix.tolist(),
        "group_counts": [len({g for g, y in zip(groups, truth, strict=True) if y == k}) for k in range(3)],
        "reference_coverage": coverage, "reviewed_recordings": len(manifest["records"]),
        "scored_clips_by_class": matrix.sum(axis=1).tolist(),
        "predictions": [{"id": ident, "group": group, "label": y, "predicted": int(pred)}
                        for ident, group, y, pred in zip(ids, groups, truth, predictions, strict=True)],
    })
    print(f"{matrix.trace()}/{matrix.sum()} correct on {manifest['split']}")


def predict_candidate(args):
    """Classify an existing center on a three-component QSIS recording."""
    from seismictorch.inference.waveform import predict_waveform
    result = predict_waveform(args.waveform, args.center, args.candidate, args.history)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit")
    for name in ("split", "review", "init", "output", "validation-manifest", "test-manifest"):
        train.add_argument(f"--{name}", required=True, type=Path)
    train.add_argument("--proposals", type=Path)
    train.add_argument("--station-review", type=Path, nargs=2, metavar=("MANIFEST", "CONSENSUS"))
    train.add_argument("--public-review", type=Path, nargs=3, metavar=("DOWNLOAD", "MANIFEST", "CONSENSUS"))
    train.add_argument("--extra-public-review", type=Path, nargs=3, action="append", default=[],
                       metavar=("DOWNLOAD", "MANIFEST", "CONSENSUS"),
                       help="Append another reviewed public training cohort, repeat for multiple cohorts")
    train.add_argument("--trigger-pool", type=Path, help="Add QSIS trigger clips from training dates")
    train.add_argument("--waveform-filter", choices=WAVEFORM_FILTERS, default="tapered")
    train.add_argument("--gravity-align", action="store_true", help="Align a stable gravity vector with the model Z axis")
    train.add_argument("--orientation-calibration", type=Path, help="Dated station calibration JSON, used with --gravity-align")
    train.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--model", choices=WAVEFORM_MODELS, default="cnn")
    train.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=20)
    score = commands.add_parser("score")
    for name in ("split", "manifest", "labels", "candidate", "output"):
        score.add_argument(f"--{name}", required=True, type=Path)
    score.add_argument("--label-provenance", required=True, help="Who reviewed the arrival labels and how")
    predict = commands.add_parser("predict")
    predict.add_argument("--candidate", type=Path, required=True)
    predict.add_argument("--waveform", type=Path, nargs=3, metavar=("X", "Y", "Z"), required=True)
    predict.add_argument("--center", type=float, required=True, help="Seconds from SAC recording start")
    predict.add_argument("--history", type=Path, help="Optional nine-second P/S history checkpoint")
    args = parser.parse_args()
    if args.command == "fit":
        if args.epochs < 1 or args.output.exists():
            parser.error("Specify positive epochs and a new output directory")
        if args.orientation_calibration and not args.gravity_align:
            parser.error("Orientation calibrations require --gravity-align")
        train_candidate(args)
    elif args.command == "score":
        score_candidate(args)
    else:
        if not math.isfinite(args.center) or args.center < 0:
            parser.error("Specify a finite nonnegative center from recording start")
        predict_candidate(args)


if __name__ == "__main__":
    main()
