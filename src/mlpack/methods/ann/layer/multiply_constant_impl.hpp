/**
 * @file multiply_constant_impl.hpp
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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MultiplyConstant<InputDataType, OutputDataType>::MultiplyConstant(
    const double scalar) : scalar(scalar)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void MultiplyConstant<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  output = input * scalar;
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void MultiplyConstant<InputDataType, OutputDataType>::Backward(
    const DataType&& /* input */, DataType&& gy, DataType&& g)
{
  g = gy * scalar;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MultiplyConstant<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(scalar);
}

} // namespace ann
} // namespace mlpack

#endif
