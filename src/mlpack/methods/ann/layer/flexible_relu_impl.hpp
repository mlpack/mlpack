/**
 * @file flexible_relu_impl.hpp
 * @author Aarush Gupta
 * @author Manthan-R-Sheth
 *
 * Implementation of FlexibleReLU layer as described by
 * Suo Qiu, Xiangmin Xu and Bolun Cai in
 * "FReLU: Flexible Rectified Linear Units for Improving Convolutional 
 *  Neural Networks", 2018
 *  
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FLEXIBLERELU_IMPL_HPP

#include "flexible_relu.hpp"
#include<algorithm>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
FlexibleReLU<InputDataType, OutputDataType>::FlexibleReLU(
    const double userAlpha) : userAlpha(userAlpha)
{
  alpha.set_size(1, 1);
  alpha(0) = userAlpha;
}

template<typename InputDataType, typename OutputDataType>
void FlexibleReLU<InputDataType, OutputDataType>::Reset()
{
  //! Set value of alpha to the one given by user.
  alpha(0) = userAlpha;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void FlexibleReLU<InputDataType, OutputDataType>::Forward(
    const InputType&& input, OutputType&& output)
{
  int i = -1;
  output = arma::zeros<InputType>(input.n_rows, input.n_cols);
  output.transform([input, &i](double val) { ++i;
      return (std::max(input(i), 0.0) + alpha); } );
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void FlexibleReLU<InputDataType, OutputDataType>::Backward(
    const DataType&& input, DataType&& gy, DataType&& g)
{
  DataType derivative;
  //! Compute the first derivative of FlexibleReLU function.
  derivative.set_size(input.n_rows, input.n_cols);
  int i = -1;
  derivative.transform([input, &i](double val) { ++i;
    return (input(i) > 0? 1 : 0); } );

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void FlexibleReLU<InputDataType, OutputDataType>::Gradient(
  const arma::Mat<eT>&& input, arma::Mat<eT>&& error,
  arma::Mat<eT>&& gradient)
{
  if (gradient.n_elem == 0)
  {
    gradient = arma::zeros<arma::Mat<eT>>(1, 1);
  }

  arma::mat zeros = arma::zeros<arma::Mat<eT>>(input.n_rows, input.n_cols);
  gradient(0) = arma::accu(error % arma::min(zeros, input)) / input.n_cols;
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
