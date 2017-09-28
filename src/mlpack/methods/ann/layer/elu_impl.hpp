/**
 * @file elu_impl.hpp
 * @author Vivek Pal
 *
 * Implementation of the ELU activation function as descibed by Djork-Arne
 * Clevert, Thomas Unterthiner and Sepp Hochreiter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ELU_IMPL_HPP

// In case it hasn't yet been included.
#include "elu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
ELU<InputDataType, OutputDataType>::ELU(
    const double alpha) : alpha(alpha)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void ELU<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  Fn(input, output);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void ELU<InputDataType, OutputDataType>::Backward(
    const DataType&& input, DataType&& gy, DataType&& g)
{
  DataType derivative;
  Deriv(input, derivative);
  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void ELU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(alpha);
}

} // namespace ann
} // namespace mlpack

#endif
