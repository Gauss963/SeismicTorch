# Historical work

`notebooks/` and `scripts/` preserve the original waveform, spectrogram,
conversion, hardware and distance-analysis work. They are historical code,
with the original hard-coded paths and dependency assumptions.

`runtime-before-reorganization.zip` is the exact root-source snapshot taken
before the package migration. Do not edit or treat it as the active codebase.
Use `seismictorch legacy-run SCRIPT.py ...` to run those archived imports and
original directory names, or `seismictorch legacy-run --workspace` to find the
persistent workspace for notebook sessions. The workspace and generated caches
are ignored by Git. New work belongs in the active package.
