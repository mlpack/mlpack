/**
 * @file methods/ann/layer/sequential_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Sequential class, which acts as a feed-forward fully
 * connected network container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SEQUENTIAL_IMPL_HPP

// In case it hasn't yet been included.
#include "sequential.hpp"

// TODO: can this be merged with MultiLayer more closely?

namespace mlpack {

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::
SequentialType() :
    reset(false), ownsLayers(true)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::
SequentialType(const bool ownsLayers) :
    reset(false), ownsLayers(ownsLayers)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::
SequentialType(const SequentialType& layer) :
    reset(layer.reset),
    ownsLayers(layer.ownsLayers)
{
  // Nothing to do here.
}

template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>&
SequentialType<InputType, OutputType, Residual>::
operator = (const SequentialType& layer)
{
  if (this != &layer)
  {
    reset = layer.reset;
    ownsLayers = layer.ownsLayers;
    network = layer.network;
    // Call copy constructor of parent...
  }
  return *this;
}


template <typename InputType, typename OutputType, bool Residual>
SequentialType<InputType, OutputType, Residual>::~SequentialType()
{
  if (ownsLayers)
  {
    for (size_t i = 0; i < network.size(); ++i)
      delete network[i];
  }
}

template <typename InputType, typename OutputType, bool Residual>
void SequentialType<InputType, OutputType, Residual>::
Forward(const InputType& input, OutputType& output)
{
  InitializeForwardPassMemory();

  network.front()->Forward(input, layerOutputs.front());

  for (size_t i = 1; i < network.size(); ++i)
  {
    network[i]->Forward(layerOutputs[i - 1], layerOutputs[i]);
  }

  // TODO: optimization is possible here
  output = layerOutputs.back();

  if (Residual)
  {
    if (arma::size(output) != arma::size(input))
    {
      Log::Fatal << "The sizes of the output and input matrices of the Residual"
          << " block should be equal. Please examine the network architecture."
          << std::endl;
    }
    output += input;
  }
}

template <typename InputType, typename OutputType, bool Residual>
void SequentialType<InputType, OutputType, Residual>::Backward(
        const InputType& /* input */,
        const OutputType& gy,
        OutputType& g)
{
  InitializeBackwardPassMemory();

  network.back()->Backward(layerOutputs.back(), gy, layerDeltas.back());

  for (size_t i = 2; i < network.size() + 1; ++i)
  {
    network[network.size() - i]->Backward(layerOutputs[network.size() - i],
        layerDeltas[network.size() - i + 1], layerDeltas[network.size() - i]);
  }

  g = layerDeltas.front();

  if (Residual)
  {
    g += gy;
  }
}

template <typename InputType, typename OutputType, bool Residual>
void SequentialType<InputType, OutputType, Residual>::
Gradient(const InputType& input,
         const OutputType& error,
         OutputType& /* gradient */)
{
  InitializeGradientPassMemory();

  network.back()->Gradient(layerOutputs[network.size() - 2], error,
      layerGradients.back());

  for (size_t i = 2; i < network.size(); ++i)
  {
    network[network.size() - i]->Gradient(
        layerOutputs[network.size() - i - 1],
        layerDeltas[network.size() - i + 1],
        layerGradients[network.size() - i]);
  }

  network.front()->Gradient(input, layerDeltas[1], layerGradients.front());
}

template <typename InputType, typename OutputType, bool Residual>
template<typename Archive>
void SequentialType<InputType, OutputType, Residual>::serialize(
        Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(ownsLayers));
}

} // namespace mlpack

#endif
