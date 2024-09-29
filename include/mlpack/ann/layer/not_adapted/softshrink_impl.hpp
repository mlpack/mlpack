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
template<typename InputType, typename OutputType>
SoftShrinkType<InputType, OutputType>::SoftShrinkType(const double lambda) :
    lambda(lambda)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void SoftShrinkType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  output = (input > lambda) % (input - lambda) +
      (input < -lambda) % (input + lambda);
}

template<typename InputType, typename OutputType>
void SoftShrinkType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  g = gy % (ones(arma::size(input)) - (input == 0));
}

template<typename InputType, typename OutputType>
template<typename Archive>
void SoftShrinkType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(lambda));
}

} // namespace mlpack

#endif
