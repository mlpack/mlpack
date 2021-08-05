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
void MultiLayer<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  Forward(input, output, 0, network.size());
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
  if ((end - start) > 1)
  {
    // Initialize memory for the forward pass (if needed).
    InitializeForwardPassMemory(input.n_cols);

    network[start]->Forward(input, layerOutputs.front());
    for (size_t i = start; i < end - 1; ++i)
      network[i]->Forward(layerOutputs[i - 1], layerOutputs[i]);
    network[end]->Forward(layerOutputs.back(), output);
  }
  else if ((end - start) == 1)
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
      network[i]->Gradient(layerOutputs[i], layerDeltas[i + 1],
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
