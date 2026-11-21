# Model artifacts

- `qsis/current/`: selected waveform CNN `model.pth`, exact training receipt,
  and original evaluation receipts. Use the directory with `seismictorch predict`.
- `legacy/`: original saved modules, TorchScript, ONNX and TensorFlow exports.
  These exports predate the current validated model.

The active model uses a state dictionary. Import its class from
`seismictorch.models.waveform`. Historical whole-module pickles retain their
original module names and can be loaded through the archived runtime.
Experiment weights and initialization sources remain in `runs/`.
