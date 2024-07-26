/**
 * @file quantization_strategy.hpp
 * @author Mark Fischinger 
 *
 * Definition of the QuantizationStrategy class, which provides an interface
 * for different quantization methods in neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_STRATEGY_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_STRATEGY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

/**
 * An abstract base class for quantization strategies. This class defines
 * the interface that all quantization strategies should implement.
 *
 * @tparam SourceMatType The source matrix type (e.g., arma::mat).
 * @tparam TargetMatType The target matrix type (e.g., arma::Mat<int8_t>).
 */
template<typename SourceMatType, typename TargetMatType>
class QuantizationStrategy
{
 public:
  /**
   * Virtual destructor for proper memory management of derived classes.
   */
  virtual ~QuantizationStrategy() = default;

  /**
   * Quantize the given weights matrix.
   *
   * @param weights The source weights to quantize.
   * @return The quantized weights.
   */
  virtual TargetMatType QuantizeWeights(const SourceMatType& weights) = 0;

  /**
   * Compute the quantization parameters for the given weights.
   * This method can be used to precompute parameters that might be
   * needed for both quantization and dequantization.
   *
   * @param weights The weights to compute parameters for.
   */
  virtual void ComputeQuantizationParameters(const SourceMatType& weights) = 0;

  /**
   * Dequantize the given quantized weights.
   *
   * @param quantizedWeights The quantized weights to dequantize.
   * @return The dequantized weights.
   */
  virtual SourceMatType DequantizeWeights(const TargetMatType& quantizedWeights) = 0;

  /**
   * Get the scale factor used in quantization.
   *
   * @return The scale factor.
   */
  virtual double GetScaleFactor() const = 0;

  /**
   * Get the zero point used in quantization (if applicable).
   *
   * @return The zero point.
   */
  virtual int GetZeroPoint() const = 0;
};

} // namespace ann
} // namespace mlpack

#endif