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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
AddType<InputType, OutputType>::AddType(const size_t outSize) :
    outSize(outSize)
{
  weights.set_size(WeightSize(), 1);
}

template<typename InputType, typename OutputType>
void AddType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  output = input;
  output.each_col() += weights;
}

template<typename InputType, typename OutputType>
void AddType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g)
{
  g = gy;
}

template<typename InputType, typename OutputType>
void AddType<InputType, OutputType>::Gradient(
    const InputType& /* input */,
    const OutputType& error,
    OutputType& gradient)
{
  gradient = error;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void AddType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(outSize));

  if (cereal::is_loading<Archive>())
    weights.set_size(outSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
