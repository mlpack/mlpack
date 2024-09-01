/**
 * @file quantization_utils.hpp
 * @author Mark Fischinger
 *
 * Utility functions and classes for quantization in neural networks.
 */
#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP

#include <mlpack/prereqs.hpp>
#include <limits>

namespace mlpack {
namespace ann {

/**
 * Base class for quantization strategies.
 */
template<typename SourceMatType, typename TargetMatType>
class QuantizationStrategy
{
 public:
  virtual TargetMatType QuantizeWeights(const SourceMatType& weights) = 0;
  virtual ~QuantizationStrategy() = default;
};

/**
 * Linear quantization strategy.
 */
class LinearQuantization : public QuantizationStrategy<arma::mat, arma::imat>
{
 public:
  LinearQuantization(size_t numBits = 8) : numBits(numBits) {}

  arma::imat QuantizeWeights(const arma::mat& weights) override
  {
    double maxAbs = arma::as_scalar(arma::max(arma::abs(weights)));
    double scale = (std::pow(2, numBits - 1) - 1) / maxAbs;
    
    arma::imat quantized = arma::conv_to<arma::imat>::from(
        arma::clamp(weights * scale, 
                    -std::pow(2, numBits - 1) + 1, 
                    std::pow(2, numBits - 1) - 1));
    
    return quantized;
  }

 private:
  size_t numBits;
};

/**
 * Scaling-based quantization strategy.
 */
class ScaleQuantization : public QuantizationStrategy<arma::mat, arma::imat>
{
 public:
  ScaleQuantization(size_t numBits = 8) : numBits(numBits) {}

  arma::imat QuantizeWeights(const arma::mat& weights) override
  {
    double maxAbs = arma::as_scalar(arma::max(arma::abs(weights)));
    double scale = maxAbs / (std::pow(2, numBits - 1) - 1);
    
    arma::imat quantized = arma::conv_to<arma::imat>::from(
        arma::clamp(arma::round(weights / scale), 
                    -std::pow(2, numBits - 1) + 1, 
                    std::pow(2, numBits - 1) - 1));
    
    return quantized;
  }

 private:
  size_t numBits;
};

/**
 * Utility function to dequantize weights.
 */
template<typename QuantizedMatType, typename FloatMatType>
FloatMatType Dequantize(const QuantizedMatType& quantizedWeights, double scale)
{
  return arma::conv_to<FloatMatType>::from(quantizedWeights) * scale;
}

/**
 * Utility function to calculate the scaling factor for quantization.
 */
template<typename MatType>
double CalculateScalingFactor(const MatType& input, const size_t numBits)
{
  double maxAbs = arma::as_scalar(arma::max(arma::abs(input)));
  return maxAbs == 0.0 ? 1.0 : maxAbs / (std::pow(2, numBits - 1) - 1);
}

} // namespace ann
} // namespace mlpack

#endif
