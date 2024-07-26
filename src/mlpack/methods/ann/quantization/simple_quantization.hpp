/**
 * @file simple_quantization.hpp
 * @author Mark Fischinger 
 *
 * Definition of the SimpleQuantization class, which implements a basic
 * linear quantization strategy for neural network weights.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_QUANTIZATION_SIMPLE_QUANTIZATION_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_SIMPLE_QUANTIZATION_HPP

#include <mlpack/prereqs.hpp>
#include "quantization_strategy.hpp"

namespace mlpack {
namespace ann {

/**
 * SimpleQuantization class implements a basic linear quantization strategy.
 * It scales the weights to fit within the range of the target data type,
 * then clamps and converts the values.
 *
 * @tparam SourceMatType The source matrix type (e.g., arma::mat).
 * @tparam TargetMatType The target matrix type (e.g., arma::Mat<int8_t>).
 */
template<typename SourceMatType, typename TargetMatType>
class SimpleQuantization : public QuantizationStrategy<SourceMatType, TargetMatType>
{
 public:
  /**
   * Constructor that initializes the quantization parameters.
   */
  SimpleQuantization() : scaleFactor(1.0), zeroPoint(0)
  {
    // Nothing to do here.
  }

  /**
   * Quantize the given weights matrix.
   *
   * @param weights The source weights to quantize.
   * @return The quantized weights.
   */
  TargetMatType QuantizeWeights(const SourceMatType& weights) override
  {
    // Ensure quantization parameters are computed.
    if (scaleFactor == 1.0)
      ComputeQuantizationParameters(weights);

    // Perform quantization.
    TargetMatType quantizedWeights;
    quantizedWeights = arma::conv_to<TargetMatType>::from(
        arma::clamp(weights * scaleFactor + zeroPoint,
                    std::numeric_limits<typename TargetMatType::elem_type>::lowest(),
                    std::numeric_limits<typename TargetMatType::elem_type>::max()));

    return quantizedWeights;
  }

  /**
   * Compute the quantization parameters for the given weights.
   *
   * @param weights The weights to compute parameters for.
   */
  void ComputeQuantizationParameters(const SourceMatType& weights) override
  {
    // Find the min and max values in the weights.
    double minVal = weights.min();
    double maxVal = weights.max();

    // Compute the scale factor and zero point.
    double targetMin = std::numeric_limits<typename TargetMatType::elem_type>::lowest();
    double targetMax = std::numeric_limits<typename TargetMatType::elem_type>::max();
    
    scaleFactor = (targetMax - targetMin) / (maxVal - minVal);
    zeroPoint = std::round(targetMin - minVal * scaleFactor);

    // Ensure zero point is within the target type's range.
    zeroPoint = std::max(static_cast<int>(targetMin), std::min(static_cast<int>(targetMax), zeroPoint));
  }

  /**
   * Dequantize the given quantized weights.
   *
   * @param quantizedWeights The quantized weights to dequantize.
   * @return The dequantized weights.
   */
  SourceMatType DequantizeWeights(const TargetMatType& quantizedWeights) override
  {
    SourceMatType dequantizedWeights;
    dequantizedWeights = arma::conv_to<SourceMatType>::from(
        (quantizedWeights - zeroPoint) / scaleFactor);

    return dequantizedWeights;
  }

  /**
   * Get the scale factor used in quantization.
   *
   * @return The scale factor.
   */
  double GetScaleFactor() const override
  {
    return scaleFactor;
  }

  /**
   * Get the zero point used in quantization.
   *
   * @return The zero point.
   */
  int GetZeroPoint() const override
  {
    return zeroPoint;
  }

 private:
  //! The scale factor used for quantization.
  double scaleFactor;

  //! The zero point used for quantization.
  int zeroPoint;
};

} // namespace ann
} // namespace mlpack

#endif