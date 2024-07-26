/**
 * @file quantize.hpp
 * @author Mark Fischinger 
 *
 * Definition of the Quantize() function for neural network quantization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include "quantization_strategy.hpp"
#include "simple_quantization.hpp"

namespace mlpack {
namespace ann {

/**
 * Quantize a neural network to use a different matrix type, typically with
 * lower precision.
 *
 * @tparam TargetMatType The desired matrix type for the quantized network.
 * @tparam NetworkType The type of the network to be quantized (FFN or RNN).
 * @tparam QuantizationStrategyType The quantization strategy to use.
 * @param network The network to be quantized.
 * @param strategy The quantization strategy object.
 * @return A new network of the same type but with quantized weights.
 */
template<
  typename TargetMatType,
  typename NetworkType,
  typename QuantizationStrategyType = SimpleQuantization<arma::mat, TargetMatType>
>
auto Quantize(
    const NetworkType& network,
    const QuantizationStrategyType& strategy = QuantizationStrategyType())
{
  return network.template Quantize<TargetMatType, QuantizationStrategyType>(strategy);
}

} // namespace ann
} // namespace mlpack

#endif