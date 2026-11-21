# C++ prototype

`SeismicTorch/` preserves the existing Xcode/LibTorch project, SAC reader and
benchmark prototype. It loads the historical TorchScript export, not the
currently validated Python checkpoint. Its waveform preprocessing and inference
pipeline are incomplete. The Xcode project also retains machine-specific
LibTorch configuration; this reorganization does not claim a working C++ build.

`fk/` is the original third-party FK seismic simulation code, with its own
README, license and sample data. It is separate from the classifier.

Run the Xcode prototype from `cpp/SeismicTorch/` when using its existing relative
paths. The maintained prediction workflow is `seismictorch predict`.
