"""Cache public QSIS event waveforms for an independent annotation audit."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.io.sac import SACTrace

from seismictorch.paths import ROOT

API = "https://smartsensors.earth.sinica.edu.tw/graphql/"
SITE = (25.03972, 121.61415)
EVENTS = """
query MyQuery($startdate: Date, $enddate: Date, $minMag: Float) {
    Events(QueryEvent: {startdate:$startdate, enddate:$enddate, minMag:$minMag}) {
        id latitude longitude date time depth ML
    }
}
"""
WAVEFORM = """
query MyQuery($code: String!, $event: String!) {
    waveforms(code: $code, event: $event) { waveform { t x y z } }
}
"""
STATIONS = """query {
    Stations { code starttime endtime building { abbreviation latitude longitude } }
}"""


def cached_triplet(paths, event, code):
    traces = [SACTrace.read(str(path), headonly=True) for path in paths]
    if len({(t.npts, t.b, t.delta) for t in traces}) != 1:
        raise ValueError(f"Mismatched cached triplet for {code}/{event}")
    return {"event": event, "station": code, "status": "downloaded",
            "paths": [str(path.relative_to(ROOT)) for path in paths],
            "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
            "samples": traces[0].npts, "start_seconds": traces[0].b, "end_seconds": traces[0].e}


def query(document, variables):
    payload = json.dumps({"query": document, "variables": variables}).encode("utf-8")
    request = Request(API, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=25) as response:
        return json.load(response)


def download(row, station, output, min_samples=40000):
    code = station["code"]
    origin = UTCDateTime(row["date"] + "T" + row["time"])
    event = origin.strftime("%Y%m%d%H%M%S")
    stamp = origin.strftime("%Y.%m.%d.%H.%M.%S")
    directory = output / "sac" / code
    paths = [directory / f"{code}.TW.HL{axis}.{stamp}.sac" for axis in "XYZ"]
    if all(path.is_file() for path in paths):
        cached = cached_triplet(paths, event, code)
        if cached["samples"] >= min_samples:
            return cached
    try:
        data = query(WAVEFORM, {"code": code, "event": event})
    except (OSError, ValueError) as error:
        return {"event": event, "station": code, "status": "request_failed", "detail": str(error)}
    waveform = ((data.get("data") or {}).get("waveforms") or {}).get("waveform")
    if not isinstance(waveform, dict) or not all(k in waveform for k in "txyz"):
        return {"event": event, "station": code, "status": "unavailable"}
    try:
        time = np.asarray(waveform["t"], dtype=float)
        xyz = np.stack([waveform[axis] for axis in "xyz"]).astype(np.float32)
    except (TypeError, ValueError):
        return {"event": event, "station": code, "status": "invalid_samples"}
    if (time.ndim != 1 or xyz.shape != (3, len(time))
            or not np.isfinite(time).all() or not np.isfinite(xyz).all()
            or not np.allclose(np.diff(time), 0.01, rtol=0, atol=1e-5)
            or (np.ptp(xyz, axis=1) == 0).any()):
        return {"event": event, "station": code, "status": "invalid_samples"}
    if len(time) < min_samples:
        return {"event": event, "station": code, "status": "short_recording", "samples": len(time)}
    directory.mkdir(parents=True, exist_ok=True)
    for axis, samples, path in zip("XYZ", xyz, paths, strict=True):
        temporary = path.with_suffix(".tmp")
        SACTrace(data=samples, delta=0.01, b=float(time[0]), o=0,
                 nzyear=origin.year, nzjday=origin.julday, nzhour=origin.hour,
                 nzmin=origin.minute, nzsec=origin.second, nzmsec=0, kstnm=code,
                 kcmpnm=f"HL{axis}", evla=row["latitude"], evlo=row["longitude"],
                 evdp=row["depth"], mag=row["ML"], stla=station["latitude"],
                 stlo=station["longitude"], kevnm=event).write(str(temporary))
        temporary.replace(path)
    return cached_triplet(paths, event, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--min-magnitude", type=float, default=4.2)
    parser.add_argument("--max-distance-km", type=float, help="Limit epicentral distance to the selected site")
    parser.add_argument("--max-depth-km", type=float)
    parser.add_argument("--site", help="Building abbreviation from the public QSIS station registry")
    parser.add_argument("--min-duration", type=int, default=400, help="Minimum recording length in seconds")
    parser.add_argument("--events", type=int, default=12, help="Maximum evenly spaced catalog events")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (args.start > args.end or not 1 <= args.events <= 24 or not output.is_relative_to(ROOT)
            or output == ROOT or (output / "download.json").exists()):
        parser.error("Use valid dates, 1–24 events and an unfinished output folder inside the repository")
    limits = (args.max_distance_km, args.max_depth_km)
    if args.min_duration < 40:
        parser.error("Recordings need at least 40 seconds for review")
    if not np.isfinite(args.min_magnitude) or any(
        limit is not None and (not np.isfinite(limit) or limit <= 0) for limit in limits
    ):
        parser.error("Use a finite magnitude and positive distance or depth limits")
    catalog = query(EVENTS, {"startdate": str(args.start), "enddate": str(args.end),
                             "minMag": args.min_magnitude})
    rows = (catalog.get("data") or {}).get("Events")
    if rows is None:
        raise ValueError(f"Public QSIS catalog query failed: {catalog.get('errors')}")
    rows = sorted((row for row in rows if row["ML"] >= args.min_magnitude
                   and str(args.start) <= row["date"] <= str(args.end)),
                  key=lambda row: (row["date"], row["time"]))
    location = SITE
    stations = [{"code": code, "latitude": SITE[0], "longitude": SITE[1]}
                for code in ("5AE83", "5AF48", "5AF0F", "5AEBA", "5AE99", "5AEE6")]
    if args.site:
        registry = query(STATIONS, {})["data"]["Stations"]
        candidates = [station for station in registry
                      if (station["building"] and station["building"]["abbreviation"] == args.site
                          and station["starttime"] and station["endtime"]
                          and station["starttime"][:10] < str(args.start)
                          and station["endtime"][:10] > str(args.end))]
        if len({station["code"] for station in candidates}) < 2:
            parser.error("Select a site with two or more instruments active throughout the dates")
        candidates.sort(key=lambda station: station["code"])
        stations = [{"code": station["code"],
                     **{key: station["building"][key] for key in ("latitude", "longitude")}}
                    for station in candidates[:6]]
        location = (stations[0]["latitude"], stations[0]["longitude"])
    if args.max_distance_km is not None:
        rows = [row for row in rows if gps2dist_azimuth(
            *location, row["latitude"], row["longitude"]
        )[0] <= args.max_distance_km * 1000]
    if args.max_depth_km is not None:
        rows = [row for row in rows if row["depth"] <= args.max_depth_km]
    if not rows:
        raise ValueError("No public QSIS events in the requested period")
    positions = np.linspace(0, len(rows) - 1, min(args.events, len(rows))).round().astype(int)
    selected = [rows[i] for i in positions]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in selected:
            event_records = list(pool.map(lambda station: download(row, station, output, args.min_duration * 100),
                                          stations))
            records.extend(event_records)
            print(row["date"], row["time"], Counter(r["status"] for r in event_records), flush=True)
    manifest = {
        "source": "https://smartsensors.earth.sinica.edu.tw/Events",
        "purpose": "Unlabeled later-period QSIS recordings for an annotation feasibility audit.",
        "origin": "Public catalog UTC to whole-second precision; t is the API relative time.",
        "selection": {"start": str(args.start), "end": str(args.end),
                      "min_magnitude": args.min_magnitude, "max_distance_km": args.max_distance_km,
                      "max_depth_km": args.max_depth_km, "site": args.site, "stations": stations,
                      "min_duration_seconds": args.min_duration},
        "events": selected, "records": records,
    }
    with (output / "download.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    print(Counter(row["status"] for row in records))


if __name__ == "__main__":
    main()
