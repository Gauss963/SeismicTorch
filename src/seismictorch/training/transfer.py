"""Test additional RED-PAN training windows without sharing audit events."""

import argparse
from collections import Counter, defaultdict
from copy import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
import torch
from torch.utils.data import TensorDataset

from seismictorch.models.waveform import WAVEFORM_MODELS
from seismictorch.training.pilot import event_samples, filtered_xyz, fit, noise_samples, trigger_noise, window
from seismictorch.qsis.public_data import public_samples
from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.training.runtime import select_device
from seismictorch.data.archives import CLASS_NAMES

JITTER_OFFSETS = (-20, 0, 20)


def recut(row, offsets, half_width=100):
    stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=700)
    if stream is None:
        raise ValueError(f"Unavailable waveform for {row['paths'][0]}")
    raw = np.stack([tr.data for tr in stream])
    filtered = filtered_xyz(stream)
    return [window(raw, filtered, row["center_sample"] + offset, half_width) for offset in offsets]


def jitter_views(baseline, half_width=100, center_offset=0):
    """Re-cut development samples from raw data rather than padding cached clips."""
    views = [[] for _ in JITTER_OFFSETS]
    valid = []
    for x, row in baseline:
        shifted = recut(row, [center_offset + offset for offset in JITTER_OFFSETS], half_width)
        if not np.array_equal(shifted[1], x):
            raise ValueError(f"Center clip differs from its recorded source: {row['id']}")
        valid.append(all(v is not None for v in shifted))
        for view, values in zip(views, shifted, strict=True):
            view.append(x if values is None else values)
    return dict(zip(JITTER_OFFSETS, map(lambda x: torch.from_numpy(np.stack(x)), views))), np.array(valid)


def extra_samples(split_path, proposals_path, audit_events, embargo=0):
    split = json.loads(split_path.read_text())
    proposals = json.loads(proposals_path.read_text())
    if proposals.get("split_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest():
        raise ValueError("Proposal split hash does not match")
    mapping = {row["paths"][0]: row for row in split["events"]}
    known = {row["group"] for row in mapping.values() if row["event_key"] in audit_events}
    if not audit_events <= {r["event_key"] for r in mapping.values()}:
        raise ValueError("Missing audited event in the frozen split")
    if embargo:
        times = {group: UTCDateTime(group).timestamp for group in {r["group"] for r in mapping.values()}}
        excluded = known | {r["group"] for r in mapping.values() if r["split"] != "train"}
        known |= {group for group, time in times.items()
                  if any(abs(time - times[holdout]) < embargo for holdout in excluded)}

    options = defaultdict(list)
    for proposed in proposals["records"]:
        row = mapping[proposed["paths"][0]]
        if row["split"] != "train" or row["group"] != proposed["group"]:
            raise ValueError("Proposal does not belong to the frozen training partition")
        if row["group"] in known:
            continue
        for label, phase in enumerate("PS"):
            pick = proposed["phases"].get(phase)
            if pick and pick["agrees"]:
                options[row["group"], label].append((min(pick["view_scores"]), pick["time"], row))

    samples = []
    for (group, label), candidates in sorted(options.items()):
        for score, time, row in sorted(candidates, key=lambda c: (-c[0], c[2]["paths"][0]))[:2]:
            stream = load_xyz([ROOT / path for path in row["paths"]])
            if stream is None:
                continue
            raw = np.stack([tr.data for tr in stream])
            center = round((row["origin_offset"] + time) * 100)
            x = window(raw, filtered_xyz(stream), center)
            if x is not None:
                samples.append((x, {"label": label, "group": group, "paths": row["paths"],
                                    "center_sample": center, "onset_center": time, "teacher_score": score}))
    return samples


def extra_noise(split_path, baseline, seed, excluded_paths=(), avoid_edges=False):
    """Sample other hard negatives per training day before forming folds."""
    original = {row["id"] for _, row in baseline if row["label"] == 2}
    days = defaultdict(list)
    for row in json.loads(split_path.read_text())["noise"]:
        if (row["split"] == "train" and Path(row["paths"][0]).name not in original
                and row["paths"][0] not in excluded_paths):
            days[row["group"]].append(row)
    rng = np.random.default_rng(seed)
    clips = []
    for day, rows in sorted(days.items()):
        used = 0
        for row in rng.permutation(rows):
            stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=700)
            clip = None if stream is None else trigger_noise(stream, min_ratio=3.01, avoid_edges=avoid_edges)
            if clip is not None:
                x, center = clip
                clips.append((x, {"label": 2, "group": day, "paths": row["paths"], "center_sample": center}))
                used += 1
            if used == 3:
                break
    return clips


def corrected_noise_samples(split_path, correction_path):
    """Reclassify reviewed earthquake onsets in recordings cataloged as noise."""
    split = json.loads(split_path.read_text())
    corrections = json.loads(correction_path.read_text())
    if corrections["split_sha256"] != hashlib.sha256(split_path.read_bytes()).hexdigest():
        raise ValueError("Noise corrections must match the frozen split")
    training = {tuple(row["paths"]): row["group"] for row in split["noise"] if row["split"] == "train"}
    evaluation_events = {row["group"] for row in split["events"] if row["split"] != "train"}
    evaluation_times = [UTCDateTime(group) for group in evaluation_events]
    clips, excluded = [], set()
    for row in corrections["records"]:
        paths = tuple(row["paths"])
        if training.get(paths) != row["noise_day"] or paths[0] in excluded or row["label"] not in (0, 1):
            raise ValueError(f"Unexpected correction for {row['id']}")
        stream = load_xyz([ROOT / path for path in paths], min_samples=700)
        if stream is None or stream[0].stats.starttime != UTCDateTime(row["record_start_utc"]):
            raise ValueError(f"Recording changed for {row['id']}")
        onset = row["onset_seconds"]
        if not np.isfinite(onset) or onset < 0:
            raise ValueError(f"Invalid arrival for {row['id']}")
        start = stream[0].stats.starttime
        origin = UTCDateTime(row["origin_utc"])
        if any(abs(origin - other) < 120 for other in evaluation_times):
            raise ValueError(f"Corrected event overlaps a held-out event near {row['id']}")
        center = round((origin - start + onset) * 100)
        raw = np.stack([trace.data for trace in stream])
        clip = window(raw, filtered_xyz(stream), center)
        if clip is None:
            raise ValueError(f"Corrected window failed QC for {row['id']}")
        clips.append((clip, {**row, "group": row["origin_utc"], "center_sample": center,
                             "source": "Catalog event with station review consensus"}))
        excluded.add(paths[0])
    return clips, excluded


def reviewed_training_samples(split_path, manifest_path, consensus_path, audit_events, embargo):
    """Use only agreed station arrivals from the frozen training partition."""
    split = json.loads(split_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    consensus = json.loads(consensus_path.read_text())
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (consensus["split"] != "train" or consensus["review_unit"] != "recording"
            or consensus["manifest_sha256"] != manifest_hash
            or any(packet["split_sha256"] != split_hash for packet in (manifest, consensus))):
        raise ValueError("Station review provenance does not match training split")
    records = {row["id"]: row for row in manifest["records"]}
    if {row["id"] for row in consensus["records"]} != records.keys():
        raise ValueError("Each station in the packet needs a review entry")
    events = {tuple(row["paths"]): row for row in split["events"]}
    excluded = {UTCDateTime(row["group"]).timestamp for row in events.values()
                if row["split"] != "train" or row["event_key"] in audit_events}
    samples = []
    used = set()
    for review in consensus["records"]:
        row = records[review["id"]]
        source = events.get(tuple(row["paths"]))
        if (source is None or source["split"] != "train"
                or any(review[key] != row[key] or row[key] != source[key]
                       for key in ("paths", "group", "event_key", "origin_offset"))):
            raise ValueError(f"Unexpected training review {row['id']}")
        time = UTCDateTime(row["group"]).timestamp
        if any(time == t or abs(time - t) < embargo for t in excluded):
            continue
        for label, phase in enumerate("ps"):
            pick = review["phases"][phase]
            if pick["status"] != "agreed":
                continue
            onset = pick["time"]
            if not np.isfinite(onset) or onset < 0:
                raise ValueError(f"Invalid station pick for {row['id']}/{phase}")
            sample = {"id": row["id"], "paths": row["paths"], "group": row["group"],
                      "label": label, "onset_center": onset, "source": "AI station consensus",
                      "center_sample": round((row["origin_offset"] + onset) * 100)}
            clip, = recut(sample, [0])
            key = (tuple(row["paths"]), label)
            if key in used:
                raise ValueError(f"Duplicate station pick for {row['id']}/{phase}")
            used.add(key)
            if clip is not None:
                samples.append((clip, sample))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "split", "proposals", "init", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--model", choices=WAVEFORM_MODELS, default="cnn")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--augment", action="store_true", help="Compare training clip augmentation")
    parser.add_argument("--jitter-eval", action="store_true",
                        help="Also evaluate ±0.2 s raw-window offsets on development folds")
    parser.add_argument("--hard-noise", action="store_true",
                        help="Compare extra STA/LTA triggers from training dates")
    parser.add_argument("--mix-noise", action="store_true",
                        help="Compare adding background noise to phase training clips")
    parser.add_argument("--raw-shifts", action="store_true",
                        help="Compare reflection padding to recutting at −0.2, 0 and +0.2 s")
    parser.add_argument("--station-review", type=Path, nargs=2, metavar=("MANIFEST", "CONSENSUS"),
                        help="Compare adding agreed training-only station arrivals")
    parser.add_argument("--public-review", type=Path, nargs=3, metavar=("DOWNLOAD", "MANIFEST", "CONSENSUS"),
                        help="Compare adding reviewed events from a later QSIS waveform archive")
    parser.add_argument("--full-rotation", action="store_true",
                        help="Compare arbitrary horizontal rotation and handedness during training")
    parser.add_argument("--noise-corrections", type=Path,
                        help="With --hard-noise, compare reviewed labels on catalog-matched recordings")
    parser.add_argument("--interior-noise", action="store_true",
                        help="Avoid edge transients when cutting additional training noise")
    parser.add_argument("--downweight-proposals", action="store_true",
                        help="Draw model-generated phase proposals half as often as reviewed clips")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Compare fitting just the final CNN layer to full fine-tuning")
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--long-context", action="store_true",
                         help="Compare 2 and 4 second CNN clips at identical centers and folds")
    context.add_argument("--later-context", action="store_true",
                         help="Compare the center window to a 2 second window starting 0.5 s before onset")
    parser.add_argument("--embargo", type=float, default=0,
                        help="Exclude proposals near evaluation event origins, in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-seed", type=int,
                        help="Change training randomness without changing the development cohort")
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1 or not np.isfinite(args.embargo) or args.embargo < 0:
        parser.error("Specify a new output directory, positive epochs and non-negative embargo")
    compare_context = args.long_context or args.later_context
    if compare_context and (args.model != "cnn" or not args.augment or args.hard_noise or args.mix_noise):
        parser.error("Context comparisons use --model cnn --augment without the other ablations")
    if args.raw_shifts and (not args.augment or compare_context or args.hard_noise or args.mix_noise):
        parser.error("Use --augment with --raw-shifts without other ablations")
    if args.station_review and (not args.augment or compare_context or args.raw_shifts or args.hard_noise or args.mix_noise):
        parser.error("Use --augment with --station-review without other ablations")
    if args.full_rotation and (not args.augment or compare_context or args.raw_shifts or args.hard_noise or args.mix_noise):
        parser.error("Use --augment with --full-rotation without other augmentation ablations")
    if args.noise_corrections and (not args.hard_noise or not args.augment):
        parser.error("--noise-corrections requires --hard-noise and --augment")
    if args.interior_noise and not args.noise_corrections:
        parser.error("--interior-noise requires --noise-corrections")
    if args.downweight_proposals and (not args.augment or compare_context or args.raw_shifts
                                     or args.hard_noise or args.mix_noise or args.full_rotation):
        parser.error("Use --augment with --downweight-proposals without other ablations")
    if args.freeze_backbone and (args.model != "cnn" or not args.augment or compare_context or args.raw_shifts
                                or args.hard_noise or args.mix_noise or args.full_rotation or args.downweight_proposals):
        parser.error("Use the augmented CNN without other model ablations")
    if args.public_review and (not args.augment or compare_context or args.raw_shifts or args.station_review
                               or args.hard_noise or args.mix_noise or args.full_rotation
                               or args.downweight_proposals or args.freeze_backbone):
        parser.error("Use --augment with --public-review without other ablations")

    with_source = args.jitter_eval or compare_context or args.raw_shifts
    event = event_samples(args.review, with_source=with_source)
    groups = sorted({row["group"] for _, _, row in event})
    noise = noise_samples(2 * len(groups), args.seed, with_source=with_source)
    baseline = [(x, row) for _, x, row in event + noise]
    jitter, jitter_valid = jitter_views(baseline) if args.jitter_eval else ({}, None)
    extra = extra_samples(args.split, args.proposals, set(groups), args.embargo)
    if not extra:
        raise ValueError("No additional independent phase proposals passed QC")
    extra_triggers = extra_noise(args.split, baseline, args.seed) if args.hard_noise else []
    corrected_triggers, corrections = [], []
    if args.noise_corrections:
        corrections, excluded = corrected_noise_samples(args.split, args.noise_corrections)
        if {Path(path).name for path in excluded} & {row["id"] for _, row in baseline if row["label"] == 2}:
            raise ValueError("A correction overlaps the fixed development cohort")
        corrected_triggers = extra_noise(args.split, baseline, args.seed, excluded, args.interior_noise)
    reviewed = []
    if args.station_review:
        reviewed = reviewed_training_samples(args.split, *args.station_review, set(groups), args.embargo)
        if not reviewed:
            raise ValueError("No independent station arrivals passed training QC")
    public = []
    if args.public_review:
        split = json.loads(args.split.read_text())
        arrivals, _, _, _ = public_samples(split, *args.public_review)
        if not arrivals:
            raise ValueError("No public station arrivals passed training QC")
        public = [(views[0], row) for views, row in arrivals]
    inputs, rows = zip(*(baseline + extra + extra_triggers + reviewed + corrected_triggers + corrections + public))
    labels = np.asarray([row["label"] for row in rows])
    dataset = TensorDataset(torch.from_numpy(np.stack(inputs)), torch.from_numpy(labels))
    context_dataset = None
    if compare_context:
        half_width = 200 if args.long_context else 100
        center_offset = 0 if args.long_context else 50
        context_key = "proposals_augmented_long_context" if args.long_context else "proposals_augmented_later_context"
        context_inputs = []
        for row in rows:
            clip, = recut(row, [center_offset], half_width)
            if clip is None:
                raise ValueError(f"Context QC failed for {row.get('id', row['paths'][0])}")
            context_inputs.append(clip)
        context_dataset = TensorDataset(torch.from_numpy(np.stack(context_inputs)), torch.from_numpy(labels))
    device = select_device(args.device)
    weights = torch.load(args.init, map_location="cpu", weights_only=True)
    settings = {"review_only": (False, False), "review_plus_proposals": (True, False)}
    if args.augment:
        settings.update(review_augmented=(False, True), proposals_augmented=(True, True))
    context_jitter = {}
    if compare_context:
        settings = {"proposals_augmented": (True, True), context_key: (True, True)}
        if jitter:
            context_baseline = list(zip(context_inputs[:len(baseline)], rows[:len(baseline)], strict=True))
            context_jitter, context_valid = jitter_views(context_baseline, half_width, center_offset)
            jitter_valid &= context_valid
    if args.hard_noise:
        settings["proposals_augmented_hard_noise"] = (True, True)
    if corrections:
        settings = {"proposals_augmented_hard_noise": (True, True),
                    "proposals_with_corrected_noise": (True, True)}
    if args.mix_noise:
        settings["proposals_augmented_noise_mix"] = (True, True)
    if args.raw_shifts:
        raw_views = []
        for x, row in zip(inputs, rows, strict=True):
            views = recut(row, JITTER_OFFSETS)
            raw_views.append(np.stack([x if view is None else view for view in views]))
        shifted_dataset = TensorDataset(torch.from_numpy(np.stack(raw_views)), torch.from_numpy(labels))
        settings = {"proposals_augmented": (True, True), "proposals_raw_shifts": (True, True)}
    if reviewed:
        settings = {"proposals_augmented": (True, True), "proposals_plus_station_review": (True, True)}
    if public:
        settings = {"proposals_augmented": (True, True), "proposals_plus_public_review": (True, True)}
    if args.full_rotation:
        settings["proposals_full_rotation"] = (True, True)
    if args.downweight_proposals:
        settings["proposals_half_sampling"] = (True, True)
        sampling_weights = torch.ones(len(rows))
        sampling_weights[len(baseline):len(baseline) + len(extra)] = 0.5
    if args.freeze_backbone:
        settings["proposals_frozen_backbone"] = (True, True)
    metrics = {key: np.zeros((3, 3), dtype=int) for key in settings}
    jitter_metrics = {offset: {key: np.zeros((3, 3), dtype=int) for key in settings} for offset in jitter}
    pooled_metrics = {key: np.zeros((3, 3), dtype=int) for key in settings} if jitter else {}
    combined_context = np.zeros((3, 3), dtype=int) if compare_context else None
    fit_args = copy(args)
    fit_args.seed = args.seed if args.fit_seed is None else args.fit_seed
    folds = []
    for i, group in enumerate(groups):
        test = [n for n, row in enumerate(rows[:len(event)]) if row["group"] == group]
        test.extend(range(len(event) + 2 * i, len(event) + 2 * i + 2))
        train = sorted(set(range(len(baseline))) - set(test))
        fold = {"event": group, "indices": test, "predictions": {}}
        if jitter:
            jitter_test = [i for i in test if jitter_valid[i]]
            fold.update(jitter_indices=jitter_test, jitter_predictions={}, pooled_predictions={})
        context_probabilities = []
        for key, (proposals, augment) in settings.items():
            alternate_context = compare_context and key == context_key
            data = context_dataset if alternate_context else dataset
            eval_views = context_jitter if alternate_context else jitter
            raw_shifts = key == "proposals_raw_shifts"
            ids = train + list(range(len(baseline), len(baseline) + len(extra))) if proposals else train
            if reviewed and key in {"proposals_plus_station_review", "proposals_full_rotation",
                                    "proposals_half_sampling", "proposals_frozen_backbone"}:
                ids += list(range(len(rows) - len(reviewed), len(rows)))
            if key == "proposals_plus_public_review":
                ids += list(range(len(rows) - len(public), len(rows)))
            if key == "proposals_augmented_hard_noise":
                held_dates = {rows[n]["group"] for n in test if rows[n]["label"] == 2}
                trigger_start = len(baseline) + len(extra)
                ids += [n for n in range(trigger_start, trigger_start + len(extra_triggers))
                        if rows[n]["group"] not in held_dates]
                fold["hard_noise_training_indices"] = ids
            if key == "proposals_with_corrected_noise":
                held_dates = {rows[n]["group"] for n in test if rows[n]["label"] == 2}
                ids += [n for n in range(len(rows) - len(corrected_triggers) - len(corrections), len(rows))
                        if rows[n].get("noise_day", rows[n]["group"]) not in held_dates]
                fold["corrected_noise_training_indices"] = ids
            bank = None
            if key == "proposals_augmented_noise_mix":
                donor_ids = [n for n in ids if labels[n] == 2]
                bank = dataset.tensors[0][donor_ids]
                fold["noise_mix_donor_indices"] = donor_ids
            model = fit(shifted_dataset if raw_shifts else data, ids, weights, device, fit_args,
                        balance=True, augment=augment, noise_bank=bank, real_shifts=raw_shifts,
                        full_rotation=key == "proposals_full_rotation",
                        example_weights=sampling_weights if key == "proposals_half_sampling" else None,
                        freeze_backbone=key == "proposals_frozen_backbone")
            with torch.inference_mode():
                probabilities = model(data.tensors[0][test].to(device)).softmax(1)
                preds = probabilities.argmax(1).cpu().numpy()
                if compare_context:
                    context_probabilities.append(probabilities.cpu().numpy())
            np.add.at(metrics[key], (labels[test], preds), 1)
            fold["predictions"][key] = preds.tolist()
            if jitter:
                fold["jitter_predictions"][key] = {}
                probabilities = []
                for offset, view in eval_views.items():
                    shifted = []
                    if jitter_test:
                        with torch.inference_mode():
                            logits = model(view[jitter_test].to(device))
                            probabilities.append(logits.softmax(1))
                            shifted = logits.argmax(1).cpu().tolist()
                        np.add.at(jitter_metrics[offset][key], (labels[jitter_test], shifted), 1)
                    fold["jitter_predictions"][key][str(offset)] = shifted
                pooled = []
                if probabilities:
                    pooled = torch.stack(probabilities).mean(0).argmax(1).cpu().tolist()
                    np.add.at(pooled_metrics[key], (labels[jitter_test], pooled), 1)
                fold["pooled_predictions"][key] = pooled
        if compare_context:
            combined = np.mean(context_probabilities, axis=0).argmax(1)
            np.add.at(combined_context, (labels[test], combined), 1)
            fold["context_probabilities"] = {
                key: probs.tolist() for key, probs in zip(settings, context_probabilities, strict=True)
            }
        folds.append(fold)
        print(f"Completed {i + 1}/{len(groups)} development folds on {device}", flush=True)

    scores = {key: {"accuracy": float(matrix.trace() / matrix.sum()),
                    "balanced_accuracy": float((matrix.diagonal() / matrix.sum(axis=1)).mean()),
                    "confusion_matrix": matrix.tolist()} for key, matrix in metrics.items()}
    summary = {"purpose": "Exploratory evaluation on previously reviewed development events.",
               "model": args.model, "device": str(device), "seed": args.seed, "epochs": args.epochs,
               "fit_seed": fit_args.seed,
               "proposal_embargo_seconds": args.embargo, "augmentation_comparison": args.augment,
               "init": str(args.init.resolve()), "classes": CLASS_NAMES,
               "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
               "proposals_sha256": hashlib.sha256(args.proposals.read_bytes()).hexdigest(),
               "additional_classes": dict(Counter(row["label"] for _, row in extra)),
               "additional_event_groups": len({row["group"] for _, row in extra}),
               "additional_hard_noise_clips": len(extra_triggers),
               "rows": rows, "folds": folds, "metrics": scores}
    if reviewed:
        summary["station_review_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in args.station_review]
        summary["station_review_clips"] = dict(Counter(row["label"] for _, row in reviewed))
    if public:
        summary["public_review_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in args.public_review]
        summary["public_review_clips"] = dict(Counter(row["label"] for _, row in public))
    if corrections:
        summary["noise_corrections_sha256"] = hashlib.sha256(args.noise_corrections.read_bytes()).hexdigest()
        summary["interior_noise"] = args.interior_noise
    if jitter:
        summary["jitter_evaluation"] = {
            "coverage": int(jitter_valid.sum()), "total": len(baseline),
            "offset_seconds": [x / 100 for x in JITTER_OFFSETS],
            "confusion_by_offset_in_samples": {
                str(offset): {key: matrix.tolist() for key, matrix in views.items()}
                for offset, views in jitter_metrics.items()
            },
            "mean_probability_confusion": {key: matrix.tolist() for key, matrix in pooled_metrics.items()},
        }
    if compare_context:
        summary["combined_context_confusion"] = combined_context.tolist()
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
