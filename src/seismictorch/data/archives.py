"""Read prepared STEAD and QSIS waveform windows."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

CLASS_NAMES = ("P-wave", "S-wave", "Noise")
MIN_AMPLITUDE = 1e-8
MAX_AMPLITUDE = 19.62


def load_waveforms(path, split):
    # The existing local archives contain NumPy arrays pickled by torch.save.
    with path.open("rb") as file:
        if file.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer. Run git lfs pull.")
    archive = torch.load(path, map_location="cpu", weights_only=False)
    waves = archive[f"X{split}"]
    one_hot = archive[f"Y{split}"]
    if waves.ndim != 4 or waves.shape[1:] != (80, 3, 1):
        raise ValueError(f"{path}: expected [N, 80, 3, 1], got {waves.shape}")
    if (one_hot.shape != (len(waves), 3) or not np.isin(one_hot, (0, 1)).all()
            or not np.all(one_hot.sum(axis=1) == 1)):
        raise ValueError(f"{path}: expected one-hot P/S/noise labels")
    if not np.isfinite(waves).all():
        raise ValueError(f"{path}: non-finite waveform samples")

    labels = one_hot.argmax(axis=1)
    groups = qsis_groups(labels) if split == "all" else None
    peak = np.abs(waves).max(axis=(1, 2, 3))
    keep = (peak > MIN_AMPLITUDE) & (peak < MAX_AMPLITUDE)
    if not keep.any():
        raise ValueError(f"{path}: no waveforms survived amplitude filtering")
    normalized = waves[keep] / peak[keep, None, None, None]
    inputs = np.ascontiguousarray(normalized.transpose(0, 3, 1, 2), dtype=np.float32)
    labels = labels[keep]
    if split == "train":
        # Duplicate STEAD windows must not land on both sides of validation.
        flat = inputs.reshape(len(inputs), -1)
        rows = flat.view(np.dtype((np.void, flat[0].nbytes))).ravel()
        _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        # If the same input has different targets, omit the whole group.
        conflicting = np.unique(inverse[labels != labels[first][inverse]])
        first = np.delete(first, conflicting)
        first.sort()
        inputs, labels = inputs[first], labels[first]
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
    return dataset, (groups[keep] if groups is not None else None), len(waves)


def qsis_groups(labels):
    # The QSIS notebooks append adjacent P/S windows from each event
    # recording, then append one noise window per noise recording.
    event_count = int(np.count_nonzero(labels != 2))
    if (event_count % 2 or not np.all(labels[:event_count:2] == 0)
            or not np.all(labels[1:event_count:2] == 1)
            or not np.all(labels[event_count:] == 2)):
        raise ValueError("QSIS archive must contain adjacent P/S pairs followed by noise")
    return np.concatenate((
        np.repeat(np.arange(event_count // 2), 2),
        np.arange(event_count // 2, len(labels) - event_count // 2),
    ))


def split_indices(labels, groups, seed):
    """Split STEAD by class and QSIS by recording type and recording ID."""
    labels = labels.numpy()
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(labels))
        strata = labels
        fractions = (0.8,)
    else:
        strata = labels == 2
        fractions = (0.7, 0.85)

    partitions = [[] for _ in range(len(fractions) + 1)]
    for stratum in np.unique(strata):
        ids = rng.permutation(np.unique(groups[strata == stratum]))
        cuts = [round(len(ids) * fraction) for fraction in fractions]
        for partition, group_ids in zip(partitions, np.split(ids, cuts)):
            partition.extend(np.flatnonzero(np.isin(groups, group_ids)))
    if any(not partition for partition in partitions):
        raise ValueError("Not enough recordings to create non-empty splits")
    return [rng.permutation(partition) for partition in partitions]
