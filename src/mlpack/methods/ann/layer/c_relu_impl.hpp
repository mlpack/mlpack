/**
 * @file methods/ann/layer/c_relu_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of CReLU layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_C_RELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_C_RELU_IMPL_HPP

// In case it hasn't yet been included.
#include "c_relu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
CReLU<InputDataType, OutputDataType>::CReLU()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void CReLU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = arma::join_cols(arma::max(input, 0.0 * input), arma::max(
      (-1 * input), 0.0 * input));
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void CReLU<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType temp;
  temp = gy % (input >= 0.0);
  g = temp.rows(0, (input.n_rows / 2 - 1)) - temp.rows(input.n_rows / 2,
      (input.n_rows - 1));
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CReLU<InputDataType, OutputDataType>::serialize(Archive& /* ar */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
