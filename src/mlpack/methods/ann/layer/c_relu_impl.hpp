/**
 * @file c_relu_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of CReLU layer.
 * Introduced by,
 * Wenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee,
 * "https://arxiv.org/abs/1603.05201", 16th March 2016.
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
    const InputType&& input, OutputType&& output)
{
  // Optimisation needed
  OutputType temp1;
  OutputType temp2;
  Fn(input, temp1);
  InputType inptemp = -1 * input;
  Fn(inptemp, temp2);
  // Concat Neg and Pos Relu
  output = arma::join_cols(temp1, temp2);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void CReLU<InputDataType, OutputDataType>::Backward(
    const DataType&& input, DataType&& gy, DataType&& g)
{
  DataType derivative;
  Deriv(input, derivative);
  DataType temp;
  temp = gy % derivative;
  g = temp.rows(0, (input.n_rows / 2 - 1)) - temp.rows(input.n_rows / 2,
                                            (input.n_rows - 1));

  /**
  * Below implementation was a different varient but couldn't manage to implement it.
  *
  * Will Clear it once Pr is done with Review
  * DataType temp1;
  * DataType temp2;
  * Deriv(input, temp1);
  * DataType inptemp=-1*input;
  * Deriv(inptemp,temp2);
  * DataType g1;
  * DataType g2;
  * g1 = gy % temp1;
  * g2 = gy % temp2;
  * derivative=arma::join_cols(temp1,temp2);
  * g=arma::join_cols(g1,g2);
  **/
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void CReLU<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
