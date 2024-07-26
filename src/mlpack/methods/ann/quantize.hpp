#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include "quantization_utils.hpp"


// In quantize.hpp
namespace mlpack {
namespace ann {

template<
    typename TargetType,
    typename NetworkType,
    typename QuantizationStrategyType = QuantizationStrategy<TargetType>
>
auto Quantize(const NetworkType& network)
{
  using SourceMatType = typename NetworkType::MatType;
  using TargetMatType = typename arma::Mat<TargetType>::template type;
  
  NetworkType quantizedNetwork;
  QuantizationStrategyType quantizationStrategy;

  // Copy network parameters
  quantizedNetwork.InputDimensions() = network.InputDimensions();
  quantizedNetwork.OutputDimensions() = network.OutputDimensions();
  quantizedNetwork.Reset() = network.Reset();
  quantizedNetwork.NumFunctions() = network.NumFunctions();

  // Quantize each layer
  for (size_t i = 0; i < network.Network().size(); ++i)
  {
    auto quantizedLayer = network.Network()[i]->template Clone<TargetMatType>();

    // Quantize weights if the layer has them
    if (quantizedLayer->Parameters().n_elem > 0)
    {
      TargetMatType quantizedWeights;
      quantizationStrategy.QuantizeWeights(network.Network()[i]->Parameters(), quantizedWeights);
      quantizedLayer->Parameters() = std::move(quantizedWeights);
    }

    quantizedNetwork.Network().push_back(quantizedLayer);
  }

  // Reset the network to ensure correct weight aliasing
  quantizedNetwork.Reset();

  return quantizedNetwork;
}

} // namespace ann
} // namespace mlpack