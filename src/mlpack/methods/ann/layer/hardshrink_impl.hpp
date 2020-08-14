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
namespace ann /** Artificial Neural Network. */ {

// This constructor is called for Hard Shrink activation function.
// 'lambda' is a hyperparameter.
template<typename InputDataType, typename OutputDataType>
HardShrink<InputDataType, OutputDataType>::HardShrink(const double lambda) :
    lambda(lambda)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void HardShrink<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = ((input > lambda) + (input < -lambda)) % input;
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void HardShrink<InputDataType, OutputDataType>::Backward(
    const DataType& input, DataType& gy, DataType& g)
{
  DataType derivative;
  derivative = (arma::ones(arma::size(input)) - (input == 0));
  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void HardShrink<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(lambda);
}

} // namespace ann
} // namespace mlpack

#endif
