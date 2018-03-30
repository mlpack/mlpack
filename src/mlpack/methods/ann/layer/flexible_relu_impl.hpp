/**
 * @file flexible_relu_impl.hpp
 * @author Aarush Gupta
 *
 * Implementation of FlexibleReLU layer as described by
 * Suo Qiu, Xiangmin Xu and Bolun Cai in
 * "FReLU: Flexible Rectified Linear Units for Improving Convolutional 
 *  Neural Networks", 2018
 *
 * For more information, read the following paper:
 *
 * @code
 * @article{
 *  author  = {Suo Qiu, Xiangmin Xu and Bolun Cai},
 *  title   = {FReLU: Flexible Rectified Linear Units for Improving
 *             Convolutional Neural Networks}
 *  journal = {arxiv preprint},
 *  year    = {2018}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_IMPL_HPP

#include "flexible_relu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
FlexibleReLU<InputDataType, OutputDataType>::FlexibleReLU(
    const double alpha) : alpha(alpha)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void FlexibleReLU<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  output = arma::max(arma::zeros<InputType>(input.n_rows, input.n_cols), input)
    + alpha;
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void FlexibleReLU<InputDataType, OutputDataType>::Backward(
    const DataType&& input, DataType&& gy, DataType&& g)
{
  DataType derivative;

  //! Compute the first derivative of FlexibleReLU function.
  derivative = input;

  for (size_t i = 0; i < input.n_elem; i++)
  {
    derivative(i) = input(i) > 0? 1 : 0;
  }

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void FlexibleReLU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version*/)
{
  ar & BOOST_SERIALIZATION_NVP(alpha);
}

} // namespace ann
} // namespace mlpack

#endif
