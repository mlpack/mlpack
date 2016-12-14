/**
 * @file hard_tanh_impl.hpp
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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
HardTanH<InputDataType, OutputDataType>::HardTanH(
    const double maxValue,
    const double minValue) :
    maxValue(maxValue),
    minValue(minValue)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void HardTanH<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  output = input;
  for (size_t i = 0; i < input.n_elem; i++)
  {
    output(i) = (output(i) > maxValue ? maxValue :
        (output(i) < minValue ? minValue : output(i)));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void HardTanH<InputDataType, OutputDataType>::Backward(
    const DataType&& input, DataType&& gy, DataType&& g)
{
  g = gy;
  for (size_t i = 0; i < input.n_elem; i++)
  {
    if (input(i) < minValue || input(i) > maxValue)
    {
      g(i) = 0;
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void HardTanH<InputDataType, OutputDataType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(maxValue, "maxValue");
  ar & data::CreateNVP(minValue, "minValue");
}

} // namespace ann
} // namespace mlpack

#endif
