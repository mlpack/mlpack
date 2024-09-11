/**
 * @file methods/ann/layer/add_merge_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of the AddMerge class, which acts as a addition container.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP

#include "add_merge.hpp"

namespace mlpack {

template<typename MatType>
AddMergeType<MatType>::AddMergeType() :
    MultiLayer<MatType>()
{
  // Nothing to do.
}

template<typename MatType>
AddMergeType<MatType>::AddMergeType(const AddMergeType& other) :
    MultiLayer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
AddMergeType<MatType>::AddMergeType(AddMergeType&& other) :
    MultiLayer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
AddMergeType<MatType>& AddMergeType<MatType>::operator=(
    const AddMergeType& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
AddMergeType<MatType>& AddMergeType<MatType>::operator=(AddMergeType&& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(std::move(other));
  }

  return *this;
}

template<typename MatType>
void AddMergeType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  // Make sure training/testing mode is set right in each layer.
  for (size_t i = 0; i < this->network.size(); ++i)
    this->network[i]->Training() = this->training;

  // Note that we use `output` for the last layer; layerOutputs is only used for
  // intermediate values between layers.
  if (this->network.size() > 1)
  {
    // Initialize temporary memory for the forward pass.
    MatType tempOutput;
    tempOutput.set_size(arma::size(output));

    // Forward pass every layer in network with same input.
    // Reduce the outputs to single output by adding element-wise.
    this->network[0]->Forward(input, tempOutput);
    output = tempOutput;
    for (size_t i = 1; i < this->network.size(); i++)
    {
      this->network[i]->Forward(input, tempOutput);
      output += tempOutput;
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
void AddMergeType<MatType>::Backward(
    const MatType& input,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  if (this->network.size() > 1)
  {
    // Initialize temporary memory for the backward pass.
    MatType tempDelta;
    tempDelta.set_size(arma::size(g));

    g.zeros();
    for (size_t i = 0; i < this->network.size(); i++)
    {
      this->network[i]->Backward(input, output, gy, tempDelta);
      g += tempDelta;
    }
  }
  else if (this->network.size() == 1)
  {
    this->network[0]->Backward(input, output, gy, g);
  }
  else
  {
    // Empty network?
    g = gy;
  }
}

template<typename MatType>
void AddMergeType<MatType>::Gradient(
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
void AddMergeType<MatType>::ComputeOutputDimensions()
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
  const std::vector<size_t>& networkSize = this->network[0]->OutputDimensions();
  for (size_t i = 1; i < this->network.size(); i++)
  {
    if (!(networkSize == this->network[i]->OutputDimensions()))
    {
      Log::Fatal << "Network size mismatch. (" << networkSize[0] << ", "
          << networkSize[1] << ") != ("
          << this->network[i]->OutputDimensions()[0] << ", "
          << this->network[i]->OutputDimensions()[1] << ")." << std::endl;
    }
  }
  this->outputDimensions = networkSize;
}

template<typename MatType>
template<typename Archive>
void AddMergeType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<MatType>>(this));
}

} // namespace mlpack

#endif
