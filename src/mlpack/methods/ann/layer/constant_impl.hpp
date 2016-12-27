/**
 * @file constant_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Constant class, which outputs a constant value given
 * any input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONSTANT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONSTANT_IMPL_HPP

// In case it hasn't yet been included.
#include "constant.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Constant<InputDataType, OutputDataType>::Constant(
    const size_t outSize,
    const double scalar) :
    inSize(0),
    outSize(outSize)
{
  constantOutput = OutputDataType(outSize, 1);
  constantOutput.fill(scalar);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void Constant<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  if (inSize == 0)
  {
    inSize = input.n_elem;
  }

  output = constantOutput;
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void Constant<InputDataType, OutputDataType>::Backward(
    const DataType&& /* input */, DataType&& /* gy */, DataType&& g)
{
  g = arma::zeros<DataType>(inSize, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Constant<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(constantOutput, "constantOutput");
}

} // namespace ann
} // namespace mlpack

#endif
