# Data layout

| Directory | Contents |
| --- | --- |
| `qsis/raw/events` | Original three-component QSIS event SAC files |
| `qsis/raw/noise` | Original QSIS noise SAC files |
| `qsis/prepared` | Historical merged QSIS PyTorch archives |
| `stead` | Prepared STEAD NPZ/PyTorch archives |
| `spectrogram/prepared` | Historical spectrogram archives |
| `samples` | Original QSIS, synthetic and spectrogram sample files |
| `reference` | Original reference material |

Directory moves preserve the original bytes. Root-level compatibility links
preserve existing training/evaluation manifest paths. Keep downloaded research
cohorts under `runs/`, with their original source and review receipts.
