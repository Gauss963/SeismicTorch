"""Locate and load synchronized 100 Hz SAC components."""

import re
from obspy import Stream, read

AXES = re.compile(r"\.(?:HL|HN)([XYZ])\.")


def triplets(directory):
    groups = {}
    for path in sorted(directory.rglob("*.sac")):
        match = AXES.search(path.name)
        if match:
            i = match.start(1)
            key = path.with_name(path.name[:i] + "?" + path.name[i + 1:])
            groups.setdefault(key, {})[match[1]] = path
    return [tuple(group[axis] for axis in "XYZ") for group in groups.values()
            if len(group) == 3]

def load_xyz(paths, min_samples=40000):
    stream = Stream([read(str(path), format="SAC")[0] for path in paths])
    first = stream[0].stats
    if first.sampling_rate != 100 or first.npts < min_samples or any(
        (tr.stats.starttime, tr.stats.sampling_rate, tr.stats.npts)
        != (first.starttime, first.sampling_rate, first.npts) for tr in stream
    ):
        return None
    return stream
