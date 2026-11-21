"""Shared waveform filtering, raw-channel QC, orientation and two-second cuts."""

import numpy as np
from scipy.signal import butter, detrend, resample, sosfilt, sosfilt_zi
from seismictorch.data.orientation import alignment

WAVEFORM_FILTERS = ("tapered", "steady-start")
HIGHPASS = butter(4, 0.5, btype="highpass", fs=100, output="sos")
HIGHPASS_STATE = sosfilt_zi(HIGHPASS)


def steady_start_filter(raw):
    """High-pass without an amplitude ramp, using the first sample to initialize the filter."""
    values = detrend(raw.astype(np.float64), axis=-1, type="linear")
    initial = HIGHPASS_STATE[:, None, :] * values[None, :, :1]
    filtered, _ = sosfilt(HIGHPASS, values, axis=-1, zi=initial)
    return filtered

def filtered_xyz(stream, mode="tapered"):
    if mode == "steady-start":
        return steady_start_filter(np.stack([trace.data for trace in stream]))
    if mode != "tapered":
        raise ValueError(f"Unknown waveform filter: {mode}")
    filtered = stream.copy().detrend("linear").taper(0.05).filter("highpass", freq=0.5)
    return np.stack([trace.data for trace in filtered])

def window(raw, filtered, center, half_width=100):
    """Cut around a sample at 100 Hz, reject dropouts, and resample to 40 Hz."""
    if not half_width <= center <= raw.shape[1] - half_width:
        return None
    section = slice(center - half_width, center + half_width)
    # A constant one-second run in any raw component invalidates the window.
    for axis in raw[:, section]:
        edges = np.r_[0, np.flatnonzero(axis[1:] != axis[:-1]) + 1, len(axis)]
        if np.diff(edges).max() >= 100:
            return None
    x = resample(filtered[:, section], half_width * 4 // 5, axis=1).T
    peak = np.abs(x).max()
    if not np.isfinite(peak) or peak == 0:
        return None
    return (x / peak)[None, :, :].astype(np.float32)


def prepared_waveform(stream, center, mode, orientation=None):
    raw = np.stack([trace.data for trace in stream])
    filtered = filtered_xyz(stream, mode)
    metadata = None
    if orientation is not None:
        if orientation.get("method") != "gravity-v1":
            raise ValueError("Unknown checkpoint orientation method")
        matrix, metadata = alignment(raw, center, stream[0].stats.station, stream[0].stats.starttime,
                                     orientation.get("calibrations", ()))
        filtered = matrix @ filtered
        metadata["rotation_xyz"] = matrix.tolist()
    return raw, filtered, metadata
