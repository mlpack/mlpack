/**
 * @file methods/ann/layer/multi_layer_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the base class for neural network layers that are wrappers
 * around other layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTI_LAYER_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTI_LAYER_IMPL_HPP

#include "multi_layer.hpp"

namespace mlpack {
namespace ann {

template<typename InputType, typename OutputType>
MultiLayer<InputType, OutputType>::MultiLayer() :
    inSize(0),
    totalInputSize(0),
    totalOutputSize(0)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
MultiLayer<InputType, OutputType>::MultiLayer(const MultiLayer& other) :
    Layer<InputType, OutputType>(other),
    inSize(other.inSize),
    totalInputSize(other.totalInputSize),
    totalOutputSize(other.totalOutputSize),
    layerOutputMatrix(other.layerOutputMatrix),
    layerDeltaMatrix(other.layerDeltaMatrix)
{
  // Copy each layer.
  for (size_t i = 0; i < other.network.size(); ++i)
    network.push_back(other.network[i]->Clone());

  // Ensure that the aliases for layers during passes have the right size.
  layerOutputs.resize(network.size(), OutputType());
  layerDeltas.resize(network.size(), OutputType());
  layerGradients.resize(network.size(), OutputType());

  // layerOutputs, layerDeltas, and layerGradients will be reset the next time
  // Forward(), Backward(), or Gradient() is called.
}

template<typename InputType, typename OutputType>
MultiLayer<InputType, OutputType>::MultiLayer(MultiLayer&& other) :
    Layer<InputType, OutputType>(other),
    network(std::move(other.network)),
    inSize(std::move(other.inSize)),
    totalInputSize(std::move(other.totalInputSize)),
    totalOutputSize(std::move(other.totalOutputSize)),
    layerOutputMatrix(std::move(other.layerOutputMatrix)),
    layerDeltaMatrix(std::move(other.layerDeltaMatrix))
{
  // Ensure that the aliases for layers during passes have the right size.
  layerOutputs.resize(network.size(), OutputType());
  layerDeltas.resize(network.size(), OutputType());
  layerGradients.resize(network.size(), OutputType());

  // layerOutputs, layerDeltas, and layerGradients will be reset the next time
  // Forward(), Backward(), or Gradient() is called.

  other.layerOutputs.clear();
  other.layerDeltas.clear();
  other.layerGradients.clear();
}

template<typename InputType, typename OutputType>
MultiLayer<InputType, OutputType>&
MultiLayer<InputType, OutputType>::operator=(const MultiLayer& other)
{
  if (this != &other)
  {
    Layer<InputType, OutputType>::operator=(other);

    network.clear();
    layerOutputs.clear();
    layerDeltas.clear();
    layerGradients.clear();

    inSize = other.inSize;
    totalInputSize = other.totalInputSize;
    totalOutputSize = other.totalOutputSize;

    layerOutputMatrix = other.layerOutputMatrix;
    layerDeltaMatrix = other.layerDeltaMatrix;

    for (size_t i = 0; i < other.network.size(); ++i)
      network.push_back(other.network[i]->Clone());

    // Ensure that the aliases for layers during passes have the right size.
    layerOutputs.resize(network.size(), OutputType());
    layerDeltas.resize(network.size(), OutputType());
    layerGradients.resize(network.size(), OutputType());
  }

  return *this;
}

template<typename InputType, typename OutputType>
MultiLayer<InputType, OutputType>&
MultiLayer<InputType, OutputType>::operator=(MultiLayer&& other)
{
  if (this != &other)
  {
    Layer<InputType, OutputType>::operator=(other);

    layerOutputs.clear();
    layerDeltas.clear();
    layerGradients.clear();

    inSize = std::move(other.inSize);
    totalInputSize = std::move(other.totalInputSize);
    totalOutputSize = std::move(other.totalOutputSize);

    // TODO: network ownerships??
    network = std::move(other.network);

    layerOutputs.resize(network.size(), OutputType());
    layerDeltas.resize(network.size(), OutputType());
    layerGradients.resize(network.size(), OutputType());

    other.layerOutputs.clear();
    other.layerDeltas.clear();
    other.layerGradients.clear();
  }

  return *this;
}

template<typename InputType, typename OutputType>
void MultiLayer<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  Forward(input, output, 0, network.size() - 1);
}

template<typename InputType, typename OutputType>
void MultiLayer<InputType, OutputType>::Forward(
    const InputType& input,
    OutputType& output,
    const size_t start,
    const size_t end)
{
  // Make sure training/testing mode is set right in each layer.
  for (size_t i = 0; i < network.size(); ++i)
    network[i]->Training() = this->training;

  // Note that we use `output` for the last layer; layerOutputs is only used for
  // intermediate values between layers.
  if ((end - start) > 0)
  {
    // Initialize memory for the forward pass (if needed).
    InitializeForwardPassMemory(input.n_cols);

    network[start]->Forward(input, layerOutputs[start]);
    for (size_t i = start + 1; i < end; ++i)
      network[i]->Forward(layerOutputs[i - 1], layerOutputs[i]);
    network[end]->Forward(layerOutputs[end - 1], output);
  }
  else if ((end - start) == 0 && network.size() > 0)
  {
    network[start]->Forward(input, output);
  }
  else
  {
    // Empty network?
    output = input;
  }
}

template<typename InputType, typename OutputType>
void MultiLayer<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  if (network.size() > 1)
  {
    // Initialize memory for the backward pass (if needed).
    InitializeBackwardPassMemory(input.n_cols);

    network.back()->Backward(input, gy, layerDeltas.back());
    for (size_t i = network.size() - 2; i > 0; --i)
      network[i]->Backward(layerOutputs[i], layerDeltas[i + 1], layerDeltas[i]);
    network[0]->Backward(layerOutputs[0], layerDeltas[1], g);
  }
  else if (network.size() == 1)
  {
    network[0]->Backward(input, gy, g);
  }
  else
  {
    // Empty network?
    g = input;
  }
}

template<typename InputType, typename OutputType>
void MultiLayer<InputType, OutputType>::Gradient(
    const InputType& input, const OutputType& error, OutputType& gradient)
{
  // We assume gradient has the right size already.

  // Pass gradients through each layer.
  if (network.size() > 1)
  {
    // Initialize memory for the gradient pass (if needed).
    InitializeGradientPassMemory(gradient);

    network.front()->Gradient(input, layerDeltas[1], layerGradients.front());
    for (size_t i = 1; i < network.size() - 1; ++i)
    {
      network[i]->Gradient(layerOutputs[i - 1], layerDeltas[i + 1],
          layerGradients[i]);
    }
    network.back()->Gradient(layerOutputs[network.size() - 2], error,
        layerGradients.back());
  }
  else if (network.size() == 1)
  {
    network[0]->Gradient(input, error, gradient);
  }
  else
  {
    // Nothing to do if the network is empty... there is no gradient.
  }
}

} // namespace ann
} // namespace mlpack

#endif
