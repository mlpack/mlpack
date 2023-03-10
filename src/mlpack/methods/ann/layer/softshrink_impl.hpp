/**
 * @file methods/ann/layer/softshrink_impl.hpp
 * @author Lakshya Ojha
 *
 * Implementation of Soft Shrink activation function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTSHRINK_IMPL_HPP

// In case it hasn't yet been included
#include "softshrink.hpp"

namespace mlpack {

// This constructor is called for Soft Shrink activation function.
// lambda is a hyperparameter.
template<typename MatType>
SoftShrinkType<MatType>::SoftShrinkType(const double lambda) :
    Layer<MatType>(),
    lambda(lambda)
{
  // Nothing to do here.
}

template<typename MatType>
SoftShrinkType<MatType>::SoftShrinkType(const SoftShrinkType& other) :
    Layer<MatType>(other),
    lambda(other.lambda)
{
  // Nothing to do here.
}

template<typename MatType>
SoftShrinkType<MatType>::SoftShrinkType(SoftShrinkType&& other) :
    Layer<MatType>(std::move(other)),
    lambda(std::move(other.lambda))
{
  // Nothing to do here.
}

template<typename MatType>
SoftShrinkType<MatType>&
SoftShrinkType<MatType>::operator=(const SoftShrinkType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    lambda = other.lambda;
  }

  return *this;
}

template<typename MatType>
SoftShrinkType<MatType>&
SoftShrinkType<MatType>::operator=(SoftShrinkType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    lambda = std::move(other.lambda);
  }

  return *this;
}

template<typename MatType>
void SoftShrinkType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = (input > lambda) % (input - lambda) +
      (input < -lambda) % (input + lambda);
}

template<typename MatType>
void SoftShrinkType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  g = gy % (arma::ones(arma::size(input)) - (input == 0));
}

template<typename MatType>
template<typename Archive>
void SoftShrinkType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(lambda));
}

} // namespace mlpack

#endif
