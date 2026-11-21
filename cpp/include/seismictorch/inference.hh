#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace seismictorch {

struct Waveform {
  double sample_rate = 100.0;
  std::int64_t start_time_ns = 0;
  std::array<std::vector<double>, 3> xyz;
  std::string stream;
  std::size_t size() const { return xyz[0].size(); }
};

struct PhaseVectors {
  std::vector<float> p;
  std::vector<float> s;
  std::vector<float> noise;
};

struct InferOptions {
  std::size_t stride = 1;
  std::size_t threads = 0; // Zero selects available CPU threads.
};

// SAC inputs are explicitly ordered X, Y, Z. miniSEED selects one complete
// NET.STATION.LOCATION.CHANNEL_PREFIX group; an ambiguous file needs a selector.
Waveform read_sac(const std::array<std::filesystem::path, 3>& files);
Waveform read_mseed(const std::filesystem::path& file, const std::string& stream = "");

// Every vector has waveform.size() elements. Unscorable edges, gaps and
// unusable windows contain quiet NaN in all three vectors, never false Noise.
PhaseVectors infer(const Waveform& waveform, InferOptions options = {});

// Lower-level entry points for embedding and numerical parity checks.
// The normalized model input is time-major [80][XYZ].
using ModelWindow = std::array<float, 240>;
std::optional<ModelWindow> prepare_window(const Waveform& waveform, std::size_t center);
std::array<float, 3> classify_window(const ModelWindow& input);
const char* model_sha256();
const char* version();

} // namespace seismictorch
