/**
 * @file methods/ann/layer/hardshrink_impl.hpp
 * @author Lakshya Ojha
 *
 * Implementation of Hard Shrink activation function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARDSHRINK_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HARDSHRINK_IMPL_HPP

// In case it hasn't yet been included
#include "hardshrink.hpp"

namespace mlpack {

// This constructor is called for Hard Shrink activation function.
// 'lambda' is a hyperparameter.
template<typename MatType>
HardShrinkType<MatType>::HardShrinkType(const double lambda) :
    Layer<MatType>(),
    lambda(lambda)
{
  // Nothing to do here.
}

template<typename MatType>
HardShrinkType<MatType>::HardShrinkType(const HardShrinkType& other) :
    Layer<MatType>(other),
    lambda(other.lambda)
{
  // Nothing to do here.
}

template<typename MatType>
HardShrinkType<MatType>::HardShrinkType(HardShrinkType&& other) :
    Layer<MatType>(std::move(other)),
    lambda(std::move(other.lambda))
{
  // Nothing to do here.
}

template<typename MatType>
HardShrinkType<MatType>&
HardShrinkType<MatType>::operator=(const HardShrinkType& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(other);
    lambda = other.lambda;
  }

  return *this;
}

template<typename MatType>
HardShrinkType<MatType>&
HardShrinkType<MatType>::operator=(HardShrinkType&& other)
{
  if (this != &other)
  {
    Layer<MatType>::operator=(std::move(other));
    lambda = std::move(other.lambda);
  }

  return *this;
}

template<typename MatType>
void HardShrinkType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = ((input > lambda) + (input < -lambda)) % input;
}

template<typename MatType>
void HardShrinkType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  g = gy % (arma::ones<MatType>(arma::size(input)) - (input == 0));
}

template<typename MatType>
template<typename Archive>
void HardShrinkType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(lambda));
}

} // namespace mlpack

#endif