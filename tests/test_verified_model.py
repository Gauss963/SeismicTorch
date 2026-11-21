"""When local evidence exists, preserve every previously scored reference prediction."""
import json
import unittest

from seismictorch.data.archives import CLASS_NAMES
from seismictorch.inference.waveform import predict_waveform
from seismictorch.paths import ROOT


def read(path):
    return json.loads(path.read_text())


def reference_cases():
    recent = ROOT / "runs/qsis-prospective-gravity"
    original = read(recent / "comparison.json")
    supplemental = read(recent / "array-followup/supplemental-comparison.json")
    phases = {r["id"]: r for r in read(recent / "phase-manifest.json")["records"]}
    noise = {r["id"]: r for r in read(recent / "background/manifest.json")["records"] if r["status"] == "rendered"}
    originals = {(r["id"], r["label"]): r for r in original["clips"]}
    additions = {(r["id"], "PS".index(r["phase"])): r for r in read(recent / "array-followup/candidates.json")["records"] if r["status"] == "additional_agreement"}
    cases = []
    for row in supplemental["clips"]:
        source = phases[row["id"]] if row["label"] != 2 else noise[row["id"]]
        if row["label"] == 2:
            center = source["center_sample"]
        else:
            key = row["id"], row["label"]
            pick = originals[key]["pick_seconds"] if key in originals else additions[key]["seconds_after_origin"]
            center = round((source["origin_offset"] + pick) * 100)
        cases.append({"id": row["id"], "paths": source["paths"], "center_sample": center, "predictions": row["predictions"]["gravity-record"]})
    for row in read(ROOT / "runs/qsis-ies-hybrid/comparison.json")["clips"]:
        cases.append({**row, "predictions": row["predictions"]["current"]})
    return cases


@unittest.skipUnless((ROOT / "runs/qsis-ies-hybrid/comparison.json").exists(), "Local benchmark evidence is not in the source distribution")
class VerifiedModelTests(unittest.TestCase):
    def test_all_fixed_reference_predictions(self):
        cases = reference_cases()
        self.assertEqual(len(cases), 47)
        for row in cases:
            for j, offset in enumerate((-20, 0, 20)):
                with self.subTest(id=row["id"], offset=offset):
                    result = predict_waveform([ROOT / path for path in row["paths"]], (row["center_sample"] + offset) / 100)
                    self.assertEqual(result["label"], CLASS_NAMES[row["predictions"][j]])


if __name__ == "__main__":
    unittest.main()
