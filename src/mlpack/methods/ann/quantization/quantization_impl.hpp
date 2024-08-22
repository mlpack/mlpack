#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_IMPL_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_IMPL_HPP

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

namespace mlpack {
namespace ann {

template<typename SourceMatType, typename TargetMatType, typename NetworkType>
NetworkType QuantizeImpl(const NetworkType& network, QuantizationStrategy<SourceMatType, TargetMatType>& strategy)
{
  NetworkType quantizedNetwork = network;
  
  for (size_t i = 0; i < network.Network().size(); ++i)
  {
    // Convert each layer in the network
    auto* layer = network.Network()[i];
    quantizedNetwork.Network()[i] = layer->template Convert<TargetMatType>();
  }

  return quantizedNetwork;
}

} // namespace ann
} // namespace mlpack

#endif
