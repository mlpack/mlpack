/**
 * @file methods/ann/layer/linear_no_bias_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LinearNoBias class also known as fully-connected layer
 * or affine transformation without the bias term.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_IMPL_HPP

// In case it hasn't yet been included.
#include "linear_no_bias.hpp"

namespace mlpack {

template<typename MatType, typename RegularizerType>
LinearNoBiasType<MatType, RegularizerType>::LinearNoBiasType() :
    Layer<MatType>(),
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
LinearNoBiasType<MatType, RegularizerType>::LinearNoBiasType(
    const size_t outSize,
    RegularizerType regularizer) :
    Layer<MatType>(),
    inSize(0), // This will be set by ComputeOutputDimensions().
    outSize(outSize),
    regularizer(regularizer)
{
  // Nothing to do.
}

template<typename MatType, typename RegularizerType>
LinearNoBiasType<MatType, RegularizerType>::LinearNoBiasType(
    const LinearNoBiasType& layer) :
    Layer<MatType>(layer),
    inSize(layer.inSize),
    outSize(layer.outSize),
    regularizer(layer.regularizer)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
LinearNoBiasType<MatType, RegularizerType>::LinearNoBiasType(
    LinearNoBiasType&& layer) :
    Layer<MatType>(std::move(layer)),
    inSize(0),
    outSize(0),
    regularizer(std::move(layer.regularizer))
{
  // Reset parameters of other layer.
  layer.inSize = 0;
  layer.outSize = 0;
}

template<typename MatType, typename RegularizerType>
LinearNoBiasType<MatType, RegularizerType>&
LinearNoBiasType<MatType, RegularizerType>::operator=(
    const LinearNoBiasType& layer)
{
  if (this != &layer)
  {
    Layer<MatType>::operator=(layer);
    inSize = layer.inSize;
    outSize = layer.outSize;
    regularizer = layer.regularizer;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
LinearNoBiasType<MatType, RegularizerType>&
LinearNoBiasType<MatType, RegularizerType>::operator=(
    LinearNoBiasType&& layer)
{
  if (this != &layer)
  {
    Layer<MatType>::operator=(std::move(layer));
    inSize = std::move(layer.inSize);
    outSize = std::move(layer.outSize);
    regularizer = std::move(layer.regularizer);

    // Reset parameters of other layer.
    layer.inSize = 0;
    layer.outSize = 0;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
void LinearNoBiasType<MatType, RegularizerType>::SetWeights(
    const MatType& weights)
{
  MakeAlias(weight, weights, outSize, inSize);
}

template<typename MatType, typename RegularizerType>
void LinearNoBiasType<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
{
  output = weight * input;
}

template<typename MatType, typename RegularizerType>
void LinearNoBiasType<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = weight.t() * gy;
}

template<typename MatType, typename RegularizerType>
void LinearNoBiasType<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  gradient.submat(0, 0, weight.n_elem - 1, 0) = vectorise(error * input.t());
  regularizer.Evaluate(weight, gradient);
}

template<typename MatType, typename RegularizerType>
void LinearNoBiasType<MatType, RegularizerType>::ComputeOutputDimensions()
{
  inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
    inSize *= this->inputDimensions[i];

  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
      1);

  this->outputDimensions[0] = outSize;
}

template<typename MatType, typename RegularizerType>
template<typename Archive>
void LinearNoBiasType<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
