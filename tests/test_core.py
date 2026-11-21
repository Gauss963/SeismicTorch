"""Small behavior checks for package boundaries and waveform invariants."""
import subprocess
import sys
import unittest

import numpy as np
import torch

from seismictorch.data.preprocessing import window
from seismictorch.data.orientation import alignment
from seismictorch.inference.history import describe
from seismictorch.models.waveform import EarthquakeCNN
from seismictorch.paths import RESOURCES, ROOT


class CoreTests(unittest.TestCase):
    def test_model_contract(self):
        model = EarthquakeCNN().eval()
        self.assertEqual([model.conv1.out_channels, model.conv2.out_channels, model.conv3.out_channels], [16, 32, 64])
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.zeros(2, 1, 80, 3)).shape), (2, 3))

    def test_raw_dropout_survives_rotation_qc(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(3, 1800))
        raw[0] += 980
        raw[1, 1100:1200] = 0
        rotation, _ = alignment(raw, 1200, "TEST", 0)
        self.assertFalse(np.allclose(rotation, np.eye(3)))
        self.assertIsNone(window(raw, rotation @ raw, 1200))
        self.assertIsNone(describe(raw, 1200, rotation=rotation))

    def test_template_is_packaged(self):
        self.assertIn("<html", (RESOURCES / "qsis_review.html").read_text().lower())

    def test_module_help_outside_workspace(self):
        result = subprocess.run([sys.executable, "-m", "seismictorch", "--help"], cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("predict", result.stdout)


if __name__ == "__main__":
    unittest.main()
