#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include "quantization_strategy.hpp"
#include "quantize_impl.hpp"

namespace mlpack {
namespace ann {

template<typename SourceMatType, typename TargetMatType, typename NetworkType>
NetworkType Quantize(const NetworkType& network, QuantizationStrategy<SourceMatType, TargetMatType>& strategy)
{
  return QuantizeImpl<SourceMatType, TargetMatType>(network, strategy);
}

} // namespace ann
} // namespace mlpack

#endif
