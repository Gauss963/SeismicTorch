"""Compare reconstructed legacy windows and visually reviewed onset windows."""

import argparse
from collections import Counter
from functools import partial
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from seismictorch.data.preprocessing import WAVEFORM_FILTERS, filtered_xyz, window
from seismictorch.models.waveform import WAVEFORM_MODELS, initialize_waveform_model
from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz, triplets
from seismictorch.training.runtime import run_epoch, select_device
from seismictorch.data.archives import CLASS_NAMES


def mix_background(x, y, bank):
    """Add training-date noise to half the phase clips at 8–20 dB RMS SNR."""
    selected = (y != 2) & (torch.rand(len(y)) < 0.5)
    count = int(selected.sum())
    if not count:
        return x
    signal = x[selected]
    noise = bank[torch.randint(len(bank), (count,))]
    # Re-center the donor independently for each component. Retain its
    # cross-axis energy ratios and scale all three axes together.
    noise = noise - noise.mean(dim=2, keepdim=True)
    signal_rms = signal.square().mean((1, 2, 3), keepdim=True).sqrt()
    noise_rms = noise.square().mean((1, 2, 3), keepdim=True).sqrt()
    snr = 8 + 12 * torch.rand(count, 1, 1, 1)
    x[selected] = signal + noise * (signal_rms / noise_rms.clamp_min(1e-8)) * 10 ** (-snr / 20)
    return x


def augment_batch(batch, noise_bank=None, shift=True, full_rotation=False):
    """Apply small time shifts and horizontal rotations to training clips."""
    x, y = torch.utils.data.default_collate(batch)
    count, _, length, _ = x.shape
    if shift:
        time = torch.arange(length) + torch.randint(-8, 9, (count, 1))
        # Reflect boundary samples instead of wrapping unrelated ends together.
        time = torch.where(time < 0, -time, time)
        time = torch.where(time >= length, 2 * (length - 1) - time, time)
        x = x.gather(2, time[:, None, :, None].expand_as(x))
    angle = (torch.rand(count, 1, 1) - 0.5) * (2 * math.pi if full_rotation else math.pi / 3)
    c, s = angle.cos(), angle.sin()
    east, north, vertical = x.unbind(-1)
    if full_rotation:
        # Some QSIS horizontal pairs have opposite handedness.
        north = north * (2 * torch.randint(2, (count, 1, 1)) - 1)
    x = torch.stack((c * east - s * north, s * east + c * north, vertical), -1)
    if noise_bank is not None:
        x = mix_background(x, y, noise_bank)
    return x / x.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8), y


def crop_shift_batch(batch):
    """Select a precomputed raw SAC cut and apply horizontal rotation."""
    views, labels = torch.utils.data.default_collate(batch)
    clips = views[torch.arange(len(views)), torch.randint(views.shape[1], (len(views),))]
    return augment_batch(list(zip(clips, labels, strict=True)), shift=False)


def event_samples(review_dir, with_source=False):
    manifest = json.loads((review_dir / "manifest.json").read_text())
    reviews = json.loads((review_dir / "ai-visual-review.json").read_text())["primary_reviews"]
    rows = {row["id"]: row for row in manifest["records"]}
    samples = []
    for review in reviews:
        if review["constant_channels"] or all(review[p + "_confidence"] != "clear" for p in "ps"):
            continue
        row = rows[review["id"]]
        if review["event_key"] != row["event_key"]:
            raise ValueError(f"Review event key mismatch for {row['id']}")
        stream = load_xyz([ROOT / path for path in row["paths"]])
        if stream is None:
            raise ValueError(f"Could not load reviewed recording {row['id']}")
        raw = np.stack([trace.data for trace in stream])
        filtered = filtered_xyz(stream)
        for label, phase in enumerate("ps"):
            limits = row["legacy"][phase.upper()]
            if review[phase + "_confidence"] != "clear" or limits is None:
                continue
            legacy_time = sum(limits) / 2
            onset_time = review[phase]
            old = window(raw, filtered, round((row["origin_offset"] + legacy_time) * 100))
            new = window(raw, filtered, round((row["origin_offset"] + onset_time) * 100))
            if old is not None and new is not None:
                sample = {
                    "id": f"{row['id']}-{phase.upper()}",
                    "group": row["event_key"], "label": label,
                    "legacy_center": legacy_time, "onset_center": onset_time,
                }
                if with_source:
                    sample.update(paths=row["paths"],
                                  center_sample=round((row["origin_offset"] + onset_time) * 100))
                samples.append((old, new, sample))
    return samples


def noise_samples(count, seed, with_source=False):
    rng = np.random.default_rng(seed)
    paths = triplets(ROOT / "data_QSIS_Noise")
    days = set()
    samples = []
    for index in rng.permutation(len(paths)):
        stream = load_xyz(paths[index], min_samples=700)
        if stream is None:
            continue
        day = str(stream[0].stats.starttime.date)
        if day in days:
            continue
        clip = trigger_noise(stream, min_ratio=3.01)
        if clip is None:
            continue
        x, center = clip
        days.add(day)
        sample = {"id": paths[index][0].name, "group": day, "label": 2}
        if with_source:
            sample.update(paths=[str(p.relative_to(ROOT)) for p in paths[index]], center_sample=center)
        samples.append((x, x, sample))
        if len(samples) == count:
            return samples
    raise ValueError(f"Only {len(samples)} independent noise days passed QC")


def trigger_noise(stream, min_ratio=0, avoid_edges=False):
    from obspy.signal.trigger import classic_sta_lta

    raw = np.stack([trace.data for trace in stream])
    if not np.isfinite(raw).all() or (np.ptp(raw, axis=1) == 0).any():
        return None
    filtered = filtered_xyz(stream)
    ratio = classic_sta_lta(np.linalg.norm(filtered, axis=0), 100, 500)
    left, right = 100, len(ratio) - 100
    if avoid_edges:
        # The 5 s LTA and the entire classification window need samples
        # outside the taper applied by filtered_xyz.
        taper = int(0.05 * len(ratio))
        left, right = taper + 500, len(ratio) - taper - 100
    if left >= right:
        return None
    center = left + int(ratio[left:right].argmax())
    if ratio[center] <= min_ratio:
        return None
    x = window(raw, filtered, center)
    return None if x is None else (x, center)


def fit(data, indices, weights, device, args, balance=False, augment=False, noise_bank=None,
        real_shifts=False, full_rotation=False, example_weights=None, freeze_backbone=False):
    torch.manual_seed(args.seed)
    model = initialize_waveform_model(args.model, weights)
    model.to(device)
    collate = None
    if real_shifts:
        collate = crop_shift_batch
    elif augment:
        collate = partial(augment_batch, noise_bank=noise_bank, full_rotation=full_rotation)
    sampler = None
    if example_weights is not None:
        # Keep the number of draws fixed and reduce the frequency of uncertain examples.
        sampler = torch.utils.data.WeightedRandomSampler(example_weights[indices], len(indices))
    loader = DataLoader(Subset(data, indices), batch_size=16, shuffle=sampler is None,
                        sampler=sampler, collate_fn=collate)
    if freeze_backbone:
        model.requires_grad_(False)
        model.fc3.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
    class_weights = None
    if balance:
        counts = data.tensors[1][indices].bincount(minlength=3)
        if (counts == 0).any():
            raise ValueError("All three classes are required for weighted training")
        class_weights = counts.float().reciprocal().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for _ in range(args.epochs):
        run_epoch(model, loader, device, criterion, optimizer, train_mode=not freeze_backbone)
    return model.eval()


def evaluate(matrix, truth, predicted):
    np.add.at(matrix, (truth, predicted), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--model", choices=WAVEFORM_MODELS, default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.output.exists():
        parser.error("Specify positive epochs and a new output directory")

    device = select_device(args.device)
    events = event_samples(args.review)
    groups = sorted({row["group"] for _, _, row in events})
    if len(groups) < 2:
        raise ValueError("Need at least two event groups")
    noise = noise_samples(2 * len(groups), args.seed)
    old, new, rows = zip(*(events + noise))
    truth = np.asarray([row["label"] for row in rows])
    targets = torch.from_numpy(truth)
    datasets = {key: TensorDataset(torch.from_numpy(np.stack(x)), targets)
                for key, x in (("legacy", old), ("onset", new))}
    weights = torch.load(args.init, map_location="cpu", weights_only=True)
    matrices = {(train_view, test_view): np.zeros((3, 3), dtype=int)
                for train_view in datasets for test_view in datasets}
    folds = []
    for i, group in enumerate(groups):
        test = [n for n, row in enumerate(rows) if row["group"] == group]
        test.extend(range(len(events) + 2 * i, len(events) + 2 * i + 2))
        train = sorted(set(range(len(rows))) - set(test))
        if set(truth[train]) != {0, 1, 2}:
            raise ValueError(f"Missing class after holding out {group}")
        fold = {"event": group, "test_indices": test, "predictions": {}}
        for train_view in datasets:
            model = fit(datasets[train_view], train, weights, device, args)
            for test_view, data in datasets.items():
                with torch.inference_mode():
                    predicted = model(data.tensors[0][test].to(device)).argmax(1).cpu().numpy()
                evaluate(matrices[train_view, test_view], truth[test], predicted)
                fold["predictions"][f"{train_view}_to_{test_view}"] = predicted.tolist()
        folds.append(fold)
        print(f"Completed held-out event {i + 1}/{len(groups)} on {device}", flush=True)

    metrics = {}
    for (train_view, test_view), matrix in matrices.items():
        recall = matrix.diagonal() / matrix.sum(axis=1)
        metrics[f"{train_view}_to_{test_view}"] = {
            "accuracy": float(matrix.trace() / matrix.sum()),
            "balanced_accuracy": float(recall.mean()),
            "recall": recall.tolist(), "confusion_matrix": matrix.tolist(),
        }
    summary = {
        "purpose": "Exploratory comparison on selected AI-reviewed picks; not deployment accuracy.",
        "model": args.model,
        "device": str(device), "seed": args.seed, "epochs": args.epochs, "lr": 3e-4,
        "init": str(args.init.resolve()), "review": str(args.review.resolve()),
        "classes": CLASS_NAMES, "class_counts": dict(Counter(truth.tolist())),
        "samples": rows, "folds": folds, "metrics": metrics,
    }
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    model = fit(datasets["onset"], list(range(len(rows))), weights, device, args)
    torch.save({key: value.cpu() for key, value in model.state_dict().items()},
               args.output / "onset-pilot.pth")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
