/**
 * @file methods/ann/layer/hard_tanh_impl.hpp
 * @author Dhawal Arora
 *
 * Implementation and implementation of the HardTanH layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_TANH_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_TANH_IMPL_HPP

// In case it hasn't yet been included.
#include "hard_tanh.hpp"

namespace mlpack {

template<typename InputType, typename OutputType>
HardTanHType<InputType, OutputType>::HardTanHType(
    const double maxValue,
    const double minValue) :
    maxValue(maxValue),
    minValue(minValue)
{
  // Nothing to do here.
}

template<typename InputType, typename OutputType>
void HardTanHType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output(i) = (output(i) > maxValue ? maxValue :
        (output(i) < minValue ? minValue : output(i)));
  }
}

template<typename InputType, typename OutputType>
void HardTanHType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  g = gy;
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (input(i) < minValue || input(i) > maxValue)
    {
      g(i) = 0;
    }
  }
}

template<typename InputType, typename OutputType>
template<typename Archive>
void HardTanHType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(maxValue));
  ar(CEREAL_NVP(minValue));
}

} // namespace mlpack

#endif
