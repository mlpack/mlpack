/**
 * @file methods/ann/layer/add_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Add class that applies a bias term to the incoming
 * data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_IMPL_HPP

// In case it hasn't yet been included.
#include "add.hpp"

namespace mlpack {

template<typename MatType>
AddType<MatType>::AddType() :
    Layer<MatType>(),
    outSize(0)
{
  // Nothing to do.
}

template<typename MatType>
AddType<MatType>::AddType(const AddType& other) :
    Layer<MatType>(other),
    outSize(other.outSize)
{
  // Nothing to do.
}

template<typename MatType>
AddType<MatType>::AddType(AddType&& other) :
    Layer<MatType>(std::move(other)),
    outSize(std::move(other.outSize))
{
  // Nothing to do.
}

template<typename MatType>
AddType<MatType>&
AddType<MatType>::operator=(const AddType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    outSize = other.outSize;
  }

  return *this;
}

template<typename MatType>
AddType<MatType>&
AddType<MatType>::operator=(AddType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    outSize = std::move(other.outSize);
  }

  return *this;
}

template<typename MatType>
void AddType<MatType>::Forward(const MatType& input, MatType& output)
{
  output = input + repmat(vectorise(weights), 1, input.n_cols);
}

template<typename MatType>
void AddType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy;
}

template<typename MatType>
void AddType<MatType>::Gradient(
    const MatType& /* input */,
    const MatType& error,
    MatType& gradient)
{
  // The gradient is the sum of the error across all input points.
  gradient = sum(error, 1);
}

template<typename MatType>
void AddType<MatType>::SetWeights(const MatType& weightsIn)
{
  // Set the weights to wrap the given memory.
  MakeAlias(weights, weightsIn, 1, outSize);
}

template<typename MatType>
void AddType<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;

  outSize = this->outputDimensions[0];
  for (size_t i = 1; i < this->outputDimensions.size(); ++i)
    outSize *= this->outputDimensions[i];
}

template<typename MatType>
template<typename Archive>
void AddType<MatType>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(weights));
}

} // namespace mlpack

#endif
