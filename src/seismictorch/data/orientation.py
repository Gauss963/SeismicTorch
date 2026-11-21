"""Align QSIS components to gravity when a stable baseline or calibration is available."""

from collections import defaultdict

import numpy as np
from obspy import UTCDateTime
from scipy.spatial.transform import Rotation

BASELINE_START = 100
BASELINE_END = 700
GRAVITY_GAL = (900, 1100)
MAX_DRIFT_GAL = 8


def gravity_direction(raw, stop=BASELINE_END):
    """Estimate the static vertical from six seconds of three-axis acceleration in gal."""
    if stop < 600:
        return None
    baseline = raw[:, stop - 600:stop].astype(np.float64)
    if baseline.shape != (3, 600) or not np.isfinite(baseline).all():
        return None
    medians = np.median(baseline.reshape(3, 6, 100), axis=-1)
    offset = np.median(medians, axis=1)
    magnitude = np.linalg.norm(offset)
    drift = np.linalg.norm(medians - offset[:, None], axis=0).max()
    if GRAVITY_GAL[0] <= magnitude <= GRAVITY_GAL[1] and drift <= MAX_DRIFT_GAL:
        return offset / magnitude
    return None


def validate_calibrations(records):
    """Validate nonoverlapping, explicitly dated calibration intervals."""
    by_station = defaultdict(list)
    for record in records:
        station = record["station"]
        start, end = (UTCDateTime(record[key]) for key in ("start_utc", "end_utc"))
        direction = np.asarray(record["up_xyz"], dtype=float)
        if (not isinstance(station, str) or not station or start >= end
                or direction.shape != (3,) or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1, rtol=0, atol=1e-6)):
            raise ValueError("Expected a station, ordered dates and a unit gravity vector")
        by_station[station].append((start, end))
    for station, dates in by_station.items():
        ordered = sorted(dates)
        if any(end > next_start for (_, end), (next_start, _) in zip(ordered, ordered[1:])):
            raise ValueError(f"Overlapping orientation calibrations for {station}")


def alignment(raw, center, station, start_time, calibrations=()):
    """Return a proper rotation and its source, using samples preceding the classifier window."""
    validate_calibrations(calibrations)
    direction = gravity_direction(raw) if BASELINE_END <= center - 100 else None
    source = "record_gravity" if direction is not None else "unchanged"
    if direction is None:
        for record in calibrations:
            if (record["station"] == station
                    and UTCDateTime(record["start_utc"]) <= start_time < UTCDateTime(record["end_utc"])):
                direction = np.asarray(record["up_xyz"], dtype=float)
                source = "station_calibration"
                break
    if direction is None:
        return np.eye(3), {"source": source, "up_xyz": None}
    rotation, _ = Rotation.align_vectors(np.array([[0.0, 0.0, 1.0]]), direction[None])
    return rotation.as_matrix(), {"source": source, "up_xyz": direction.tolist()}
