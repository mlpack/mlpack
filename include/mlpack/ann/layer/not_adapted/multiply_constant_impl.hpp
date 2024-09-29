/**
 * @file methods/ann/layer/multiply_constant_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the MultiplyConstantLayer class, which multiplies the
 * input by a (non-learnable) constant.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_CONSTANT_IMPL_HPP

// In case it hasn't yet been included.
#include "multiply_constant.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
MultiplyConstantType<InputType, OutputType>::MultiplyConstantType(
    const double scalar) : scalar(scalar)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void MultiplyConstantType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  output = input * scalar;
}

template<typename InputType, typename OutputType>
void MultiplyConstantType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  g = gy * scalar;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void MultiplyConstantType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(scalar));
}

} // namespace mlpack

#endif
