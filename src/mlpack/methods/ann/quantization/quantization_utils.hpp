/**
 * @file quantization_utils.hpp
 * @author Mark Fischinger 
 *
 * Utility functions for quantization in neural networks.
 */
#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

/**
 * Perform linear quantization on the input data.
 *
 * @param input The input data to quantize.
 * @param minVal The minimum value of the quantization range.
 * @param maxVal The maximum value of the quantization range.
 * @param numBits The number of bits to use for quantization.
 * @return The quantized data.
 */
template<typename MatType>
MatType LinearQuantize(const MatType& input,
                       const double minVal,
                       const double maxVal,
                       const size_t numBits)
{
  // Ensure valid range
  if (minVal >= maxVal)
  {
    throw std::invalid_argument("minVal must be less than maxVal.");
  }

  // Calculate the scale factor
  const double scale = (std::pow(2, numBits) - 1) / (maxVal - minVal);

  // Perform quantization
  MatType quantized = arma::floor((input - minVal) * scale + 0.5);

  // Clamp values to the valid range
  const double maxQuantizedVal = std::pow(2, numBits) - 1;
  quantized = arma::clamp(quantized, 0.0, maxQuantizedVal);

  // Scale back to the original range
  return (quantized / scale) + minVal;
}

/**
 * Calculate the scaling factor for quantization.
 *
 * @param input The input data to calculate the scaling factor for.
 * @param numBits The number of bits to use for quantization.
 * @return The scaling factor.
 */
template<typename MatType>
double CalculateScalingFactor(const MatType& input, const size_t numBits)
{
  // Prevent division by zero if input is zero
  const double maxAbs = arma::abs(input).max();
  if (maxAbs == 0.0)
  {
    return 1.0;  // Avoid division by zero, though input is already effectively zero.
  }

  return maxAbs / (std::pow(2, numBits - 1) - 1);
}

/**
 * Perform scaling-based quantization on the input data.
 *
 * @param input The input data to quantize.
 * @param numBits The number of bits to use for quantization.
 * @return The quantized data.
 */
template<typename MatType>
MatType ScaleQuantize(const MatType& input, const size_t numBits)
{
  const double scale = CalculateScalingFactor(input, numBits);

  // Perform quantization
  MatType quantized = arma::round(input / scale);

  // Clamp values to the valid range
  const double maxVal = std::pow(2, numBits - 1) - 1;
  quantized = arma::clamp(quantized, -maxVal, maxVal);

  // Scale back to the original range
  return quantized * scale;
}

} // namespace ann
} // namespace mlpack

#endif