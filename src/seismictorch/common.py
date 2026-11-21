"""Checksums and immutable JSON receipts shared by the QSIS workflow."""

import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path, data):
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
