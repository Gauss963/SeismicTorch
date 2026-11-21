"""Evaluate a frozen U-Trans encoder with a small grouped linear classifier."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz


def extract_context(row):
    stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=6000)
    if stream is None:
        return None
    center = row["center_sample"]
    start = center - 3000
    if start < 0 or center + 3000 > stream[0].stats.npts:
        return None
    wave = np.stack([t.data[start:center + 3000] for t in stream], axis=1).astype(np.float32)
    if not np.isfinite(wave).all() or (wave.std(axis=0) <= 1e-8).any():
        return None
    return wave


def probe(train, train_labels, test):
    """Fit a regularized three-class linear head with fold-local normalization."""
    scale = np.maximum(train.std(axis=0), 1e-5)
    mean = train.mean(axis=0)
    train = (train - mean) / scale
    test = (test - mean) / scale
    targets = np.eye(3)[train_labels]
    counts = targets.sum(axis=0)
    if not (counts > 0).all():
        raise ValueError("Each training fold must have all three classes")
    weights = (1 / counts[train_labels]) / 3

    def objective(params):
        w = params[:-3].reshape(-1, 3)
        logits = train @ w + params[-3:]
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        grad = (np.exp(logp) - targets) * weights[:, None]
        loss = -np.sum(weights * logp[np.arange(len(train)), train_labels]) + 0.5 * np.sum(w * w)
        return loss, np.r_[(train.T @ grad + w).ravel(), grad.sum(axis=0)]

    result = minimize(objective, np.zeros(train.shape[1] * 3 + 3), jac=True, method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Linear head did not converge: {result.message}")
    return (test @ result.x[:-3].reshape(-1, 3) + result.x[-3:]).argmax(axis=1)


def evaluate(model, data, normalize_std):
    rows = data["rows"][:data["jitter_evaluation"]["total"]]
    clips = [extract_context(row) for row in rows]
    present = np.array([x is not None for x in clips])
    if not present.any():
        raise ValueError("No usable U-Trans contexts")
    inputs = normalize_std(np.stack([x for x in clips if x is not None]))
    tokens = np.concatenate([np.asarray(model(x, training=False))
                             for x in np.array_split(inputs, (len(inputs) + 7) // 8)])
    if tokens.shape[1:] != (75, 80) or not np.isfinite(tokens).all():
        raise ValueError(f"Unexpected U-Trans embeddings: {tokens.shape}")
    # Pool over fixed regions before, around, and after the candidate onset.
    pooled = np.concatenate([t.mean(axis=1) for t in (tokens[:, 30:36], tokens[:, 36:39], tokens[:, 39:45])], axis=1)
    features = np.zeros((len(rows), pooled.shape[1]))
    features[present] = pooled
    labels = np.asarray([row["label"] for row in rows])
    matrix, baseline = np.zeros((3, 3), dtype=int), np.zeros((3, 3), dtype=int)
    folds = []
    for fold in data["folds"]:
        ids = fold["indices"]
        mask = np.zeros(len(rows), dtype=bool)
        mask[ids] = True
        train, test = present & ~mask, present & mask
        if not test.any():
            continue
        pred = probe(features[train], labels[train], features[test])
        np.add.at(matrix, (labels[test], pred), 1)
        prior = fold["predictions"]["review_only"]
        filtered = [(labels[i], p) for i, p in zip(ids, prior, strict=True) if present[i]]
        np.add.at(baseline, tuple(zip(*filtered, strict=True)), 1)
        folds.append({"event": fold["event"], "indices": np.flatnonzero(test).tolist(),
                      "predictions": pred.tolist()})
    return {
        "purpose": "Development comparison using selected AI visual references.",
        "input": "60 s around the review onset or noise trigger; 100 Hz",
        "coverage": int(present.sum()), "total": len(rows),
        "class_counts": dict(Counter(labels[present].tolist())),
        "accuracy": float(matrix.trace() / matrix.sum()), "confusion_matrix": matrix.tolist(),
        "cnn_review_only_same_subset": float(baseline.trace() / baseline.sum()),
        "cnn_confusion_matrix_same_subset": baseline.tolist(),
        "excluded": [row["id"] for row, exists in zip(rows, present, strict=True) if not exists],
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "weights", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Specify a new output path")

    import tensorflow as tf
    from utrans.foundation import get_latent_model
    from utrans.preprocessing import normalize_std

    tf.config.threading.set_intra_op_parallelism_threads(4)
    model = get_latent_model(str(args.weights))
    model.trainable = False
    data = json.loads(args.development.read_text())
    result = evaluate(model, data, normalize_std)
    result["weights_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    result["development_sha256"] = hashlib.sha256(args.development.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print({k: v for k, v in result.items() if k not in {"excluded", "folds"}})


if __name__ == "__main__":
    main()
