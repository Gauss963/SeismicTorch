"""Score a possible P precursor using nine seconds of waveform history."""

import numpy as np
import torch

RATE = 100
HISTORY_SECONDS = 9
FEATURES = 20
INPUT_SECONDS = HISTORY_SECONDS + 1
MAX_LOG_ODDS = 1.25
TAPER = np.hanning(RATE)


def describe(raw, center, rotation=None):
    """Compute band energy and vertical fraction, given a 100 Hz XYZ stream."""
    start = center - HISTORY_SECONDS * RATE
    finish = center + RATE
    if not 0 <= start < finish <= raw.shape[1]:
        return None
    samples = raw[:, start:finish].astype(np.float64)
    if not np.isfinite(samples).all():
        return None
    for axis in samples:
        edges = np.r_[0, np.flatnonzero(np.diff(axis)) + 1, len(axis)]
        if np.diff(edges).max() >= RATE:
            return None
    # Check the acquired components before rotation can mix a repeated
    # sensor channel with valid signal from the other channels.
    if rotation is not None:
        samples = np.asarray(rotation, dtype=np.float64) @ samples
    bins = samples.reshape(3, INPUT_SECONDS, RATE)
    tapered = (bins - bins.mean(axis=-1, keepdims=True)) * TAPER
    spectra = np.abs(np.fft.rfft(tapered, axis=-1)) ** 2
    energies = np.stack([spectra[..., 1:5].sum(axis=-1), spectra[..., 5:19].sum(axis=-1)])
    total = energies.sum(axis=1)
    floor = np.maximum(total.max(axis=1) * 1e-8, np.finfo(float).tiny)
    background = np.maximum(np.median(np.sort(total[:, :8], axis=1)[:, :3], axis=1), floor)
    level = np.clip(np.log(np.maximum(total, floor[:, None]) / background[:, None]), -4, 8)
    vertical = energies[:, 2] / np.maximum(total, floor[:, None])
    result = []
    # A broad precursor can precede the central window by several seconds.
    for selection in (slice(0, 4), slice(4, 8), slice(8, 9), slice(9, 10)):
        result.extend((level[:, selection].max(axis=1), vertical[:, selection].mean(axis=1)))
    result.extend((level[:, 8] - level[:, 7], level[:, 9] - level[:, 8]))
    return np.concatenate(result).astype(np.float32)


def refine(logits, context_log_odds, available, max_log_odds=MAX_LOG_ODDS):
    """Keep the CNN's noise decision and refine its conditional P/S decision."""
    original = logits.argmax(dim=-1)
    prior = context_log_odds.clamp(-max_log_odds, max_log_odds)
    is_s = logits[..., 1] - logits[..., 0] + prior > 0
    phase = is_s.to(dtype=original.dtype)
    return torch.where(available & (original != 2), phase, original)


def predict(logits, features, checkpoint, max_log_odds=MAX_LOG_ODDS):
    """Refine [batch, 3] CNN logits using one optional context vector per clip."""
    if logits.shape != (len(features), 3):
        raise ValueError("Expected one history vector per set of P, S and noise logits")
    available = torch.tensor([x is not None for x in features], device=logits.device)
    filled = [x if x is not None else np.zeros(FEATURES, np.float32) for x in features]
    values = torch.as_tensor(np.stack(filled), device=logits.device, dtype=logits.dtype)
    mu, std = (checkpoint[key].to(values) for key in ("mean", "std"))
    weight, bias = (checkpoint["state"][key].to(values) for key in ("weight", "bias"))
    odds = torch.nn.functional.linear((values - mu) / std, weight, bias).squeeze(-1)
    return refine(logits, odds, available, max_log_odds)
