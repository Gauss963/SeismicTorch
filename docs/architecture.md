# Code organization

## Active code

The package has one command dispatcher, `seismictorch.cli`. It imports a
workflow only when requested, so ordinary inference does not import plotting,
training experiments, TensorFlow or external pickers.

| Area | Responsibility |
| --- | --- |
| `models.waveform` | The validated CNN and optional comparison architectures |
| `data.sac` | Synchronization checks and SAC XYZ discovery/loading |
| `data.archives` | Prepared STEAD/QSIS archive validation and splitting |
| `data.preprocessing` | Shared high-pass filtering, raw-channel QC and window cuts |
| `data.orientation` | Per-record gravity and explicitly dated calibrations |
| `training.runtime` | Device selection and epoch execution |
| `training.supervised` | Prepared-archive pretraining/fine-tuning |
| `training.cwa`, `training.pilot`, `training.transfer` | CWA adaptation and reviewed QSIS experiments |
| `inference.waveform`, `inference.history` | Classifier prediction and optional history refinement |
| `qsis.candidate` | Fitting and scoring a frozen QSIS candidate |
| Other `qsis` modules | Acquisition, review packets, consensus, splits and diagnostics |
| `resources` | HTML shipped with the package |

Data loading and inference share the same preprocessing implementation. Keep
plotting and optional picker dependencies outside that path. Add new research
runners under their own `runs/` directory; promote reusable behavior into the
appropriate package module after validation.

## Data and evidence

Raw QSIS files live in `data/qsis/raw/{events,noise}`. Prepared QSIS arrays live
in `data/qsis/prepared`; STEAD archives in `data/stead`; historical spectrogram
arrays in `data/spectrogram/prepared`. Files were moved within the same volume,
with the directory inode preserved. No dataset content was regenerated.

Four root-level directory links retain the original identifiers used by
immutable manifests: `data_QSIS_Event`, `data_QSIS_Noise`, `data_QSIS_pth` and
`data_train`. Removing these links would break older relative paths in the
receipts. There is one physical copy of each dataset.

`models/qsis/current` contains exact copies of the selected checkpoint and its
receipts. The source experiment stays in `runs/qsis-ccul-extra/gravity-record-candidate`.
Training receipts and all prior `runs/` evidence retain their original bytes.
Migration parity checks are recorded separately under `runs/repository-reorganization`.

## Historical code

The original root Python files, review HTML and documentation were archived
before refactoring. `docs/reorganization.json` records their SHA-256 values,
new module names and directory moves. The ZIP is a source snapshot rather than
a second maintained implementation.

`seismictorch legacy-run` materializes that exact snapshot under
`.cache/legacy-runtime/`, then links the original directory names to their
current locations. It runs in a separate command process, with the archived
imports ahead of active package code. This preserves original class names for
historical saved modules and original source hashes for experiment replay.
New files written directly in that workspace persist there; `runs/` outputs
are linked back to the repository's `runs/`. Old scripts can still require
obsolete dependencies, absolute external paths or unavailable services. New
acquisition, training and prediction should use the active package commands.

The original notebooks remain readable under `legacy/notebooks/{waveform,spectrogram}`.
Historical scripts are grouped by waveform, spectrogram, export, hardware and
distance analysis. Original figures and reports live under `reports/legacy`.

## Commands

| Previous root script | Active command |
| --- | --- |
| `train.py` | `seismictorch train` |
| `qsis_candidate.py` | `seismictorch qsis` |
| `qsis_candidate.py predict` | `seismictorch predict` |
| `qsis_review.py` | `seismictorch tools review` |
| `qsis_split.py` | `seismictorch tools split` |
| `qsis_reference.py` | `seismictorch tools reference` |
| `qsis_array_review.py` | `seismictorch tools array-review` |
| `qsis_array_proposals.py` | `seismictorch tools array-proposals` |
| `qsis_consensus.py` | `seismictorch tools consensus` |
| `qsis_public.py` | `seismictorch tools public` |
| `qsis_public_score.py` | `seismictorch tools public-score` |
| `qsis_catalog.py` | `seismictorch tools catalog` |
| `qsis_noise_audit.py` | `seismictorch tools noise-audit` |
| `qsis_redpan.py` | `seismictorch tools redpan` |
| `qsis_pilot.py` | `seismictorch tools pilot` |
| `qsis_transfer.py` | `seismictorch tools transfer` |
| `qsis_cwa.py` | `seismictorch tools cwa` |
| `qsis_utrans.py` | `seismictorch tools utrans` |

The CLI argument meanings, architecture, augmentation and numerical training
settings remain the same. Reorganization does not promote any experimental
model or finish the separate C++ inference implementation.

## Reorganization verification

The migrated fitting command reproduced all 226 ordered training windows and
all 27 state tensors bit-for-bit on MPS. The resulting checkpoint SHA-256 is
identical to the selected original model. The five regression tests include
47 existing reference windows at all three offsets, preserving all 141
predictions, including known errors. All 19 command help routes were checked.
A built wheel includes the review HTML and performs prediction from outside
the checkout. Frozen-runtime checks verify all 43 archived source hashes.

The structured checks are in `docs/reorganization-verification.json`; local
training and wheel details are under `runs/repository-reorganization/`. The
Git index and commits were left untouched. Directory moves appear as deleted
old paths plus new destinations in unstaged status until the changes are
staged; the dataset files themselves remain present under `data/`.
