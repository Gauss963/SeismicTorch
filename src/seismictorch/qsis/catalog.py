"""Find public nearby-station ISC arrival context for QSIS validation events."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import urlopen

from obspy import UTCDateTime, read
from obspy.geodetics import gps2dist_azimuth

from seismictorch.paths import ROOT

ISC_URL = "https://www.isc.ac.uk/cgi-bin/web-db-run"
PHASES = {"P", "Pg", "Pn", "S", "Sg", "Sn"}


def distance(a, b):
    return gps2dist_azimuth(*a, *b)[0] / 1000


def query_day(day, cache):
    path = cache / f"isc-{day}.txt"
    if path.exists():
        text = path.read_text()
    else:
        year, month, date = day.split("-")
        query = dict(out_format="CSV", request="STNARRIVALS", stnsearch="CIRC",
                     stn_ctr_lat=25.05, stn_ctr_lon=121.6, max_stn_dist_units="deg",
                     stn_radius=0.55, phaselist="P,Pg,Pn,S,Sg,Sn", searchshape="GLOBAL",
                     start_year=year, start_month=month, start_day=date, start_time="00:00:00",
                     end_year=year, end_month=month, end_day=date, end_time="23:59:59")
        url = ISC_URL + "?" + urlencode(query)
        with urlopen(url, timeout=25) as response:
            text = response.read().decode()
        if "cannot be processed" in text:
            raise RuntimeError("ISC returned a transient processing error")
        path.write_text(text)
    rows = (line for line in text.splitlines() if re.match(r"^\s*\d+\s*,", line))
    events = defaultdict(list)
    for fields in csv.reader(rows):
        if len(fields) < 28:
            continue
        r = [value.strip() for value in fields]
        phase = r[11] if r[11] in PHASES else r[12]
        if phase not in PHASES or not all(r[i] for i in (5, 6, 13, 14, 20, 21, 22, 23)):
            continue
        events[r[0]].append({
            "station": r[3], "station_coords": [float(r[5]), float(r[6])],
            "agency": r[2], "phase": phase[0],
            "arrival_utc": str(UTCDateTime(r[13] + "T" + r[14])),
            "origin_utc": str(UTCDateTime(r[20] + "T" + r[21])),
            "epicenter": [float(r[22]), float(r[23])],
        })
    return events


def context_for(record, catalog):
    origin = UTCDateTime(record["origin_utc"])
    *_, lat, lon = record["event_key"].split("/")
    epicenter = float(lat), float(lon)
    station = read(str(ROOT / record["paths"][0]), format="SAC", headonly=True)[0].stats.sac
    site = float(station.stla), float(station.stlo)
    choices = []
    for ident, arrivals in catalog.items():
        item = arrivals[0]
        dt = abs(UTCDateTime(item["origin_utc"]) - origin)
        km = distance(epicenter, item["epicenter"])
        if dt <= 5 and km <= 30:
            choices.append((dt, km, ident, arrivals))
    if len(choices) != 1:
        return {"id": record["id"], "status": "ambiguous" if choices else "unmatched"}
    dt, km, ident, arrivals = choices[0]
    closest = {}
    for phase in "PS":
        unique = {}
        for arrival in arrivals:
            if arrival["phase"] == phase:
                unique.setdefault(arrival["station"], arrival)
        ranked = sorted(unique.values(), key=lambda row: distance(site, row["station_coords"]))
        closest[phase] = [
            {**row, "distance_to_qsis_km": distance(site, row["station_coords"]),
             "seconds_from_qsis_origin": UTCDateTime(row["arrival_utc"]) - origin}
            for row in ranked[:3]
        ]
    return {"id": record["id"], "status": "nearby_station_context", "isc_event_id": ident,
            "origin_difference_seconds": dt, "epicenter_difference_km": km, "nearest": closest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["split"] != "validation" or args.output.exists():
        parser.error("Use a validation manifest and a new output path")
    cache = args.output.parent / "isc-cache"
    cache.mkdir(parents=True, exist_ok=True)
    dates = sorted({row["origin_utc"][:10] for row in manifest["records"]})
    catalogs, failed = {}, {}
    for day in dates:
        try:
            catalogs[day] = query_day(day, cache)
        except (OSError, ValueError, RuntimeError) as error:
            failed[day] = str(error)
        print(f"{day}: {len(catalogs.get(day, {}))} ISC events", flush=True)
    records = [context_for(row, catalogs.get(row["origin_utc"][:10], {}))
               for row in manifest["records"]]
    result = {
        "source": "International Seismological Centre Bulletin (https://www.isc.ac.uk/iscbulletin/)",
        "purpose": "Nearby station context; timings are not QSIS reference labels.",
        "split_sha256": manifest["split_sha256"], "failed_queries": failed, "records": records,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Matched {sum('isc_event_id' in row for row in records)}/{len(records)} recordings")


if __name__ == "__main__":
    main()
