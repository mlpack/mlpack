/**
 * @file methods/ann/layer/flexible_relu_impl.hpp
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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
FlexibleReLUType<InputType, OutputType>::FlexibleReLUType(const double alpha) :
    userAlpha(alpha)
{
  this->alpha.set_size(1, 1);
  this->alpha(0) = userAlpha;
}

template<typename InputType, typename OutputType>
void FlexibleReLUType<InputType, OutputType>::Reset()
{
  alpha = arma::mat(alpha.memptr(), 1, 1, false, false);

  // Set value of alpha to the one given by user.
  alpha(0) = userAlpha;
}

template<typename InputType, typename OutputType>
void FlexibleReLUType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  output = arma::clamp(input, 0.0, DBL_MAX) + alpha(0);
}

template<typename InputType, typename OutputType>
void FlexibleReLUType<InputType, OutputType>::Backward(
    const InputType& input, const OutputType& gy, OutputType& g)
{
  // Compute the first derivative of FlexibleReLU function.
  g = gy % arma::clamp(arma::sign(input), 0.0, 1.0);
}

template<typename InputType, typename OutputType>
void FlexibleReLUType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  if (gradient.n_elem == 0)
    gradient.set_size(1, 1);

  gradient(0) = arma::accu(error) / input.n_cols;
}

template<typename InputType, typename OutputType>
template<typename Archive>
void FlexibleReLUType<InputType, OutputType>::serialize(
    Archive& ar,
    const uint32_t /* version*/)
{
  ar(CEREAL_NVP(alpha));
}

} // namespace ann
} // namespace mlpack

#endif
