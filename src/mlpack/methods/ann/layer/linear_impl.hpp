/**
 * @file methods/ann/layer/linear_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Linear layer class also known as fully-connected layer
 * or affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP

// In case it hasn't yet been included.
#include "linear.hpp"

namespace mlpack {

template<typename MatType, typename RegularizerType>
LinearType<MatType, RegularizerType>::LinearType() :
    Layer<MatType>(),
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename MatType, typename RegularizerType>
LinearType<MatType, RegularizerType>::LinearType(
    const size_t outSize,
    RegularizerType regularizer) :
    Layer<MatType>(),
    inSize(0), // This will be computed in ComputeOutputDimensions().
    outSize(outSize),
    regularizer(regularizer)
{
  // Nothing to do here.
}

// Copy constructor.
template<typename MatType, typename RegularizerType>
LinearType<MatType, RegularizerType>::LinearType(const LinearType& layer) :
    Layer<MatType>(layer),
    inSize(layer.inSize),
    outSize(layer.outSize),
    regularizer(layer.regularizer)
{
  // Nothing else to do.
}

// Move constructor.
template<typename MatType, typename RegularizerType>
LinearType<MatType, RegularizerType>::LinearType(LinearType&& layer) :
    Layer<MatType>(std::move(layer)),
    inSize(std::move(layer.inSize)),
    outSize(std::move(layer.outSize)),
    regularizer(std::move(layer.regularizer))
{
  // Reset parameters of other layer.
  layer.inSize = 0;
  layer.outSize = 0;
}

template<typename MatType, typename RegularizerType>
LinearType<MatType, RegularizerType>&
LinearType<MatType, RegularizerType>::operator=(const LinearType& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(layer);
    inSize = layer.inSize;
    outSize = layer.outSize;
    regularizer = layer.regularizer;
  }

  return *this;
}

template<typename MatType, typename RegularizerType>
LinearType<MatType, RegularizerType>&
LinearType<MatType, RegularizerType>::operator=(
    LinearType&& layer)
{
  if (&layer != this)
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
void LinearType<MatType, RegularizerType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, outSize * inSize + outSize, 1);
  MakeAlias(weight, weightsIn, outSize, inSize);
  MakeAlias(bias, weightsIn, outSize, 1, weight.n_elem);
}

template<typename MatType, typename RegularizerType>
void LinearType<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
{
  output = weight * input;

  #pragma omp for
  for (size_t c = 0; c < (size_t) output.n_cols; ++c)
    output.col(c) += bias;
}

template<typename MatType, typename RegularizerType>
void LinearType<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = weight.t() * gy;
}

template<typename MatType, typename RegularizerType>
void LinearType<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  gradient.submat(0, 0, weight.n_elem - 1, 0) = vectorise(error * input.t());
  gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) = sum(error, 1);
  regularizer.Evaluate(weights, gradient);
}

template<typename MatType, typename RegularizerType>
void LinearType<MatType, RegularizerType>::ComputeOutputDimensions()
{
  inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
    inSize *= this->inputDimensions[i];
  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
      1);

  // The Linear layer flattens its input.
  this->outputDimensions[0] = outSize;
}

template<typename MatType, typename RegularizerType>
template<typename Archive>
void LinearType<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
