# Standalone CPU inference

This C++17/CMake project embeds the selected QSIS CNN directly in the executable.
Normal compilation needs a C/C++ compiler and CMake, with **no Python, model
file, LibTorch, MLX or separately installed miniSEED library**. The vendored
Apache-2.0 libmseed 3.6.0 decoder is linked statically, with URL/curl support off.

## Build

```sh
cmake -S cpp -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release --parallel
ctest --test-dir build/native -C Release --output-on-failure
```

On Windows use a Visual Studio developer environment; the CRT is linked with
`/MT`. On macOS the binary uses only Apple's system C/C++ libraries. For a
universal Mac executable add `'-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64'` and
`-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0`. Linux release CI builds with musl and
`-DSEISMICTORCH_STATIC=ON`, then verifies there is no dynamic interpreter.
Ordinary Linux builds use the host's standard runtime unless this flag is set.

Our C++ files use `.cc` and `.hh`; unmodified third-party C retains `.c`/`.h`.
The old Xcode/LibTorch prototype and FK files are archived in `legacy/cpp/`.

## Command line

```sh
seismictorch-infer --sac X.sac Y.sac Z.sac --output probabilities.csv
seismictorch-infer --mseed recording.mseed --output probabilities.csv
seismictorch-infer --mseed multiple-stations.mseed --stream XX.TEST..HL
```

Three positional SAC paths or one positional miniSEED path also work. The CSV
has `time_seconds,P,S,Noise`, with exactly one row per sample, relative to the
start of the recording. No output path writes CSV to stdout.

`--stride 1` is the default: infer every supported sample center. Larger strides
explicitly select an approximation, using linear interpolation between valid
anchors. `--threads N` controls CPU workers; zero selects available cores.
`--version` prints the program version and embedded checkpoint SHA-256.

## C++ interface

```cpp
#include <seismictorch/inference.hh>

auto wave = seismictorch::read_sac({"X.sac", "Y.sac", "Z.sac"});
// Or: auto wave = seismictorch::read_mseed("recording.mseed");
auto scores = seismictorch::infer(wave);
std::vector<float> P = std::move(scores.p);
std::vector<float> S = std::move(scores.s);
std::vector<float> Noise = std::move(scores.noise);
```

Each vector has `wave.size()` elements. Add this directory with CMake
`add_subdirectory()` and link `SeismicTorch::seismictorch` to embed the API in
another application. The only public interface types are standard C++ types.

## Input and output semantics

- SAC: binary v6/v7, little or big endian, evenly sampled time series. X/Y/Z
  files must have equal sample rates, UTC starts and sample counts.
- miniSEED: versions 2 and 3, including Steim1/2 and numeric float/integer
  encodings supported by libmseed. One complete network/station/location/channel
  family is selected. XYZ, ENZ and 12Z suffixes map to the three axes. Multiple
  complete families require `--stream NET.STATION.LOCATION.CHANNEL_PREFIX`.
- miniSEED segments use a common UTC sample grid over their full time span;
  gaps or missing component samples are NaN. Conflicting overlaps are rejected.
- Inputs must be acceleration waveforms. The gravity alignment check expects
  gal (cm/s²); the program does not remove instrument response or infer units.
- Sample rates from 40 to 1000 Hz are accepted. Each two-second window is
  Fourier-resampled to the model's 80 samples per axis. The 100 Hz route matches
  the verified Python pipeline. Native-rate high-pass coefficients are used
  at other rates.
- Full finite segments are linearly detrended and high-pass filtered at 0.5 Hz
  with steady initial conditions. Stable gravity from seconds 1–7 is aligned
  vertically only when it precedes the classified window. Original acquired
  axes are checked for one-second repeated runs before rotation.
- A center needs the complete two-second window. Unsupported edges, gaps,
  repeated-channel windows or zero signals return **NaN in all three vectors**.
  They are not fabricated Noise predictions. Gaps restart segment processing.
- Finite P/S/Noise entries are softmax probabilities summing to one. These are
  sliding-window classifier probabilities, not a newly trained sequence picker.
- Reader limits are 2 GB per file and 50 million samples on the aligned grid.

## Model and verification

The embedded checkpoint is
`634eea951ff2f8e4a42f4bbf12174aaa0a8aa6699d15726d4cba9e8cf90fe424`.
It uses convolutions 16/32/64 and FC 384/64/16/3. Batch normalization is folded
into the convolution constants; exact GELU and softmax run on CPU.

`generated/model_weights.hh` is committed, so consumers do not regenerate it.
To reproduce it in the existing Python training environment:

```sh
python cpp/tools/export_weights.py
python cpp/tools/generate_fixtures.py
python cpp/tools/check_parity.py
```

Tests include 141 real held-out model windows, complete preprocessing golden
cases, SAC endianness, compressed miniSEED 2/3, CRC rejection, Unicode paths,
raw dropout QC, dense vector lengths and gap/interpolation behavior. The full
local parity script checks actual SAC decoding plus preprocessing and inference
against the unchanged Python model, without shipping the large datasets to CI.

GitHub Actions builds/tests Ubuntu, macOS and Windows before publishing all
three packages on a `v*` tag. Packages include license notices and one standalone
inference executable; the notices are not runtime dependencies.
