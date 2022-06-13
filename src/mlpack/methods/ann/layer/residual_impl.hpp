/**
 * @file methods/ann/layer/residual_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of the base class for neural network layers that are wrappers
 * around other layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RESIDUAL_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RESIDUAL_IMPL_HPP

#include "residual.hpp"

namespace mlpack {
namespace ann {

template<typename MatType>
ResidualType<MatType>::ResidualType() :
    MultiLayer<MatType>()
{
  // Nothing to do.
}

template<typename MatType>
ResidualType<MatType>::ResidualType(const ResidualType& other) :
    MultiLayer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
ResidualType<MatType>::ResidualType(ResidualType&& other) :
    MultiLayer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
ResidualType<MatType>& ResidualType<MatType>::operator=(const ResidualType& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
ResidualType<MatType>& ResidualType<MatType>::operator=(ResidualType&& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
void ResidualType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  // Make sure training/testing mode is set right in each layer.
  for (size_t i = 0; i < this->network.size(); ++i)
    this->network[i]->Training() = this->training;

  // Note that we use `output` for the last layer; layerOutputs is only used for
  // intermediate values between layers.
  if (this->network.size() > 1)
  {
    // Initialize memory for the forward pass (if needed).
    this->InitializeForwardPassMemory(input.n_cols);

    for (size_t i = 0; i < this->network.size(); i++)
      this->network[i]->Forward(input, this->layerOutputs[i]);

		// Reduce the outputs to single output.
    output.zeros();
		for (size_t i = 0; i < this->layerOutputs.size(); i++)
		{
			output += this->layerOutputs[i];
		}
  }
  else if (this->network.size() == 1)
  {
    this->network[0]->Forward(input, output);
  }
  else
  {
    // Empty network?
    output = input;
  }
}

template<typename MatType>
void ResidualType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  if (this->network.size() > 1)
  {
    // Initialize memory for the backward pass (if needed).
    this->InitializeBackwardPassMemory(input.n_cols);

    g.zeros();
    for (size_t i = 0; i < this->network.size(); i++) {
      this->network[i]->Backward(this->layerOutputs[i], gy, this->layerDeltas[i]);
			g += this->layerDeltas[i];
		}
  }
  else if (this->network.size() == 1)
  {
    this->network[0]->Backward(input, gy, g);
  }
  else
  {
    // Empty network?
    g = gy;
  }
}

template<typename MatType>
void ResidualType<MatType>::Gradient(
    const MatType& input, const MatType& error, MatType& gradient)
{
  // We assume gradient has the right size already.

  // Pass gradients through each layer.
  if (this->network.size() > 1)
  {
    // Initialize memory for the gradient pass (if needed).
    this->InitializeGradientPassMemory(gradient);

    for (size_t i = 0; i < this->network.size(); ++i)
    {
      this->network[i]->Gradient(input, error, this->layerGradients[i]);
    }
  }
  else if (this->network.size() == 1)
  {
    this->network[0]->Gradient(input, error, gradient);
  }
  else
  {
    // Nothing to do if the network is empty... there is no gradient.
  }
}

template<typename MatType>
void ResidualType<MatType>::ComputeOutputDimensions()
{
  this->inSize = 0;
  this->totalInputSize = 0;
  this->totalOutputSize = 0;

  // Propagate the input dimensions forward to the output.
  if (this->network.size() == 0)
  {
    this->outputDimensions = this->inputDimensions;
    return;
  }
  this->inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
    this->inSize *= this->inputDimensions[i];
  this->totalInputSize = this->network.size() * this->inSize;

  for (size_t i = 0; i < this->network.size(); ++i)
  {
    this->network[i]->InputDimensions() = this->inputDimensions;
    size_t layerOutputSize = this->network[i]->OutputSize();
    this->totalOutputSize += layerOutputSize;
  }

	// Compute the output size of the network using reduction rules.
  if (this->network.size() == 1) 
  {
    this->outputDimensions = this->network[0]->OutputDimensions();
    return;
  }
  const std::vector<size_t> networkSize = this->network[0]->OutputDimensions();
  for (size_t i = 1; i < this->network.size(); i++)
  {
    if (!(networkSize == this->network[i]->OutputDimensions()))
    {
      Log::Fatal << "Network size mismatch. (" << networkSize[0] << ", "
        << networkSize[1] << ") != ("
        << this->network[i]->OutputDimensions()[0] << ", " << this->network[i]->OutputDimensions()[1]
        << ")." << std::endl;
    }
  }
  this->outputDimensions = networkSize;
}

template<typename MatType>
template<typename Archive>
void ResidualType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<MatType>>(this));
}

} // namespace ann
} // namespace mlpack

#endif
