"""Find QSIS noise triggers near CWA catalog arrivals."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

from seismictorch.training.pilot import trigger_noise
from seismictorch.qsis.public import STATIONS, query
from seismictorch.paths import ROOT
from seismictorch.data.sac import load_xyz
from seismictorch.qsis.review import travel_times

def catalog(path, years):
    events = []
    with zipfile.ZipFile(path) as archive:
        for year in sorted(years):
            name = f"CWA-EQ-Catalog-{year}.xml"
            for row in ET.fromstring(archive.read(name)).findall(".//{*}EarthquakeInfo"):
                origin = row.findtext("{*}OriginTime")
                fields = [row.findtext(f"{{*}}{key}") for key in (
                    "EpicenterLatitude", "EpicenterLongitude", "FocalDepth", "LocalMagnitude"
                )]
                if origin and all(fields):
                    events.append((UTCDateTime(origin), *map(float, fields)))
    return events


def run(split_path, catalog_path, output, partitions=("train", "validation")):
    split = json.loads(split_path.read_text())
    rows = [row for row in split["noise"] if row["split"] in partitions]
    events = catalog(catalog_path, {row["group"][:4] for row in rows})
    by_station = defaultdict(list)
    registry = query(STATIONS, {})["data"]["Stations"]
    codes = {Path(row["paths"][0]).name.split(".")[0] for row in rows}
    for station in registry:
        if station["code"] in codes and station["building"] and station["starttime"] and station["endtime"]:
            by_station[station["code"]].append({
                **station, "start": UTCDateTime(station["starttime"]), "end": UTCDateTime(station["endtime"])
            })
    taup = TauPyModel("ak135")
    coverage = defaultdict(int)
    matches = []
    for row in rows:
        stream = load_xyz([ROOT / path for path in row["paths"]], min_samples=700)
        trigger = trigger_noise(stream) if stream is not None else None
        if trigger is None:
            coverage["invalid_trigger"] += 1
            continue
        time = stream[0].stats.starttime + trigger[1] / 100
        code = Path(row["paths"][0]).name.split(".")[0]
        sites = [station["building"] for station in by_station[code]
                 if station["start"] <= time < station["end"]]
        if len(sites) != 1:
            coverage["unknown_station_site"] += 1
            continue
        coverage["checked"] += 1
        site = sites[0]
        candidates = []
        for origin, lat, lon, depth, magnitude in events:
            elapsed = time - origin
            if not -10 <= elapsed <= 180 or magnitude < 2:
                continue
            distance = gps2dist_azimuth(site["latitude"], site["longitude"], lat, lon)[0] / 1000
            if distance > 400:
                continue
            _, expected = travel_times(taup, {"distance_km": distance, "depth_km": depth})
            offsets = {phase: round(elapsed - value, 3) for phase, value in expected.items()}
            if min(map(abs, offsets.values())) <= 5:
                candidates.append({"origin_utc": str(origin), "magnitude": magnitude,
                                   "distance_km": round(distance, 2), "trigger_minus_arrival_seconds": offsets})
        if candidates:
            matches.append({**row, "station": code, "site": site["abbreviation"],
                            "trigger_utc": str(time), "candidates": candidates})
    result = {
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "station_registry": {code: [{key: station[key] for key in ("starttime", "endtime", "building")}
                                    for station in stations] for code, stations in by_station.items()},
        "catalog_events": len(events),
        "scope": "Arrival proximity is a review lead.",
        "partitions": list(partitions),
        "coverage": dict(coverage), "matches": matches,
    }
    with output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Found {len(matches)} candidate triggers. Coverage: {dict(coverage)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split", "catalog", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--partition", choices=("train", "validation", "test"), action="append",
                        help="Noise partitions to inspect. Defaults to train and validation.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new audit output file")
    run(args.split, args.catalog, args.output, args.partition or ("train", "validation"))
