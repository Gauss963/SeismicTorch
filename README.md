# SeismicTorch

Three-component waveform classification for P waves, S waves and noise. The
current CNN uses two-second windows at 40 Hz, convolution channels
`16 → 32 → 64`, and fully connected dimensions `384 → 64 → 16 → 3`.

## Repository layout

```text
src/seismictorch/       Active Python package
  models/              Waveform model definitions
  data/                SAC/archive loading, filtering, QC and orientation
  training/            Training runtime, STEAD/CWA adaptation and QSIS pilots
  inference/           Prediction and optional waveform-history refinement
  qsis/                Review, event splits, proposals and evaluation tools
  resources/           Offline arrival-review HTML
models/qsis/current/   Verified checkpoint, training receipt and evaluation
models/legacy/         Historical saved modules and exports
data/                  Raw QSIS, prepared datasets, STEAD and reference samples
cpp/                   Existing C++ prototype and FK source
legacy/                Original notebooks, scripts and frozen source snapshot
reports/legacy/        Historical figures, reports and slides
docs/                  Benchmark results, commands and architecture notes
tests/                 Behavior and migration regression checks
runs/                  Local experiment outputs and immutable evidence (ignored)
```

The four `data_QSIS_*` / `data_train` links at the root preserve recorded
paths in existing split manifests and receipts. Their actual files live under
`data/`. New application code belongs under `src/seismictorch/`.

## Install

From the repository root, activate a Python environment with PyTorch. The
existing local `.venv/qsis-review` environment uses the verified Python 3.12
runtime:

```sh
source .venv/qsis-review/bin/activate
python -m pip install -e '.[training,review]'
seismictorch --help
```

`python -m seismictorch` is the equivalent entry point. Prediction needs only
the base dependencies. TensorFlow and external pickers are optional research
dependencies, installed with `.[pickers]` when needed. Downloaded picker weights
and experimental datasets stay in `runs/`.

## Predict with the verified QSIS model

```sh
seismictorch predict --center 126.0 --waveform X.sac Y.sac Z.sac
```

Inputs are synchronized **100 Hz** SAC files in X, Y, Z order. The center is
seconds from recording start. The checkpoint's training receipt determines
filtering and gravity alignment. The default is `models/qsis/current`; use
`--candidate OTHER_DIRECTORY` to select another checkpoint/receipt pair.
The output includes the label, checkpoint hash and preprocessing metadata.

```python
from seismictorch.inference.waveform import predict_waveform

result = predict_waveform(["X.sac", "Y.sac", "Z.sac"], center_seconds=126.0)
print(result["label"])
```

The retained model scored **46/47 (97.87%)** at center across the two independent
QSIS confirmation cohorts including three supplemental array references.
The weakest fixed ±0.2-second result was **45/47 (95.74%)**. References came
from independent model-blind AI waveform reviews. Unresolved arrival slots
remain outside the score and are explicitly counted in the
[benchmark report](docs/qsis-benchmark.md#current-verified-result).

## Train and review

The [verified QSIS reproduction command](docs/qsis-benchmark.md#reproduce-the-gravity-candidate)
uses event/date separation, CWA initialization, reviewed arrivals and
training-only proposals/noise. It reproduces the retained 226-source model.
For the prepared-archive workflow:

```sh
seismictorch train pretrain --device mps --output runs/stead
seismictorch train finetune --device mps --init runs/stead/best.pth --output runs/qsis
```

`--device auto` prefers CUDA, then Apple MPS, then CPU. The prepared QSIS archive
lacks full event identifiers; its split is by recording. The reviewed QSIS
workflow above provides the stronger event/date evaluation.

```sh
seismictorch tools review --output runs/qsis-review --count 100
seismictorch tools --help
seismictorch qsis fit --help
```

Use the offline review page to record candidate arrival times. Pickers propose
arrivals for review. The classifier predicts an existing two-second window;
continuous arrival detection is a separate task.

## Historical work and development

Original notebooks are grouped under `legacy/notebooks/`; scripts are grouped
under `legacy/scripts/`. The original root Python sources are also preserved
byte-for-byte in `legacy/runtime-before-reorganization.zip`, so frozen source
hashes and older saved-model class names remain available.

```sh
# Replay an old script with its archived imports and original data layout.
seismictorch legacy-run qsis_candidate.py --help
# Locate the persistent historical workspace for notebook work.
seismictorch legacy-run --workspace
```

Use the active commands for new work. Historical scripts retain their original
dependencies and limitations. Experiment evidence in `runs/` has not been
rewritten. See [the migration notes](docs/architecture.md) for paths and scope.
The C++ code remains a prototype; see [its status](cpp/README.md).

```sh
python -m unittest discover -s tests -v
```

Set `SEISMICTORCH_ROOT` when using an installed package with a workspace located
elsewhere. The existing local prepared archives use Git LFS and contain trusted
pickled NumPy data; obtain their contents with `git lfs pull` when cloning.
