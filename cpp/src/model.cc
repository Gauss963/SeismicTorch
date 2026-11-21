#include "seismictorch/inference.hh"
#include "model_weights.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace seismictorch {
namespace {
float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475244f)); }

template <std::size_t InChannels, std::size_t OutChannels, std::size_t Time>
auto conv_pool(const std::array<float, InChannels * Time * 3>& input,
               const std::array<float, OutChannels * InChannels * 3>& weights,
               const std::array<float, OutChannels>& bias) {
  constexpr auto OutputTime = (Time - 2) / 3;
  std::array<float, OutChannels * OutputTime * 3> output{};
  for (std::size_t out = 0; out < OutChannels; ++out) {
    for (std::size_t t = 0; t < OutputTime; ++t) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        float best = -INFINITY;
        for (std::size_t pool = 0; pool < 3; ++pool) {
          float value = bias[out];
          for (std::size_t in = 0; in < InChannels; ++in)
            for (std::size_t k = 0; k < 3; ++k)
              value += input[(in * Time + t * 3 + pool + k) * 3 + axis] *
                       weights[(out * InChannels + in) * 3 + k];
          best = std::max(best, gelu(value));
        }
        output[(out * OutputTime + t) * 3 + axis] = best;
      }
    }
  }
  return output;
}

template <std::size_t In, std::size_t Out>
auto dense(const std::array<float, In>& input, const std::array<float, In * Out>& weights,
           const std::array<float, Out>& bias, bool activation) {
  std::array<float, Out> output{};
  for (std::size_t out = 0; out < Out; ++out) {
    float value = bias[out];
    for (std::size_t in = 0; in < In; ++in) value += input[in] * weights[out * In + in];
    output[out] = activation ? gelu(value) : value;
  }
  return output;
}
} // namespace

std::array<float, 3> classify_window(const ModelWindow& input) {
  for (auto x : input)
    if (!std::isfinite(x)) throw std::invalid_argument("Model input must be finite");
  auto x = conv_pool<1, 16, 80>(input, embedded::conv1_weight, embedded::conv1_bias);
  auto y = conv_pool<16, 32, 26>(x, embedded::conv2_weight, embedded::conv2_bias);
  auto z = conv_pool<32, 64, 8>(y, embedded::conv3_weight, embedded::conv3_bias);
  auto a = dense<384, 64>(z, embedded::fc1_weight, embedded::fc1_bias, true);
  auto b = dense<64, 16>(a, embedded::fc2_weight, embedded::fc2_bias, true);
  auto logits = dense<16, 3>(b, embedded::fc3_weight, embedded::fc3_bias, false);
  float maximum = *std::max_element(logits.begin(), logits.end());
  float sum = 0;
  for (auto& x : logits) { x = std::exp(x - maximum); sum += x; }
  for (auto& x : logits) x /= sum;
  return logits;
}

const char* model_sha256() { return embedded::checkpoint_sha256; }
const char* version() { return embedded::version; }
} // namespace seismictorch
