/**
 * @file celu_impl.hpp
 * @author Gaurav Singh
 *
 * Implementation of the CELU activation function as descibed by Jonathan T. Barron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CELU_IMPL_HPP

// In case it hasn't yet been included.
#include "celu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
CELU<InputDataType, OutputDataType>::CELU(const double alpha) :
    alpha(alpha),
    deterministic(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void CELU<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  Fn(input, output);

  if(!deterministic)
  {
    Deriv(input, output);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void CELU<InputDataType, OutputDataType>::Backward(
    const DataType&& /* input */, DataType&& gy, DataType&& g)
{
  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CELU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(alpha);
}
    
} // namespace ann
} // namespace mlpack

#endif