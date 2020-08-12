/**
 * @file methods/ann/layer/parametric_relu_impl.hpp
 * @author Prasanna Patil
 *
 * Definition of PReLU layer first introduced in the,
 * Kaiming He, Xiangyu Zhang, Shaoqing, Ren Jian Sun,
 * "Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification", 2014
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_PReLU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_PReLU_IMPL_HPP

// In case it hasn't yet been included.
#include "parametric_relu.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PReLU<InputDataType, OutputDataType>::PReLU(
    const double userAlpha) : userAlpha(userAlpha)
{
  alpha.set_size(1, 1);
  alpha(0) = userAlpha;
}

template<typename InputDataType, typename OutputDataType>
void PReLU<InputDataType, OutputDataType>::Reset()
{
  //! Set value of alpha to the one given by user.
  alpha(0) = userAlpha;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void PReLU<InputDataType, OutputDataType>::Forward(
    const InputType& input, OutputType& output)
{
  output = input;
  arma::uvec negative = arma::find(input < 0);
  output(negative) = input(negative) * alpha(0);
}

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void PReLU<InputDataType, OutputDataType>::Backward(
    const DataType& input, const DataType& gy, DataType& g)
{
  DataType derivative;
  derivative.set_size(arma::size(input));
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    derivative(i) = (input(i) >= 0) ? 1 : alpha(0);
  }

  g = gy % derivative;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void PReLU<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  if (gradient.n_elem == 0)
  {
    gradient = arma::zeros<arma::mat>(1, 1);
  }

  arma::mat zeros = arma::zeros<arma::mat>(input.n_rows, input.n_cols);
  gradient(0) = arma::accu(error % arma::min(zeros, input)) / input.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PReLU<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(alpha);
}

} // namespace ann
} // namespace mlpack

#endif
