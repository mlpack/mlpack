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

#ifndef MLPACK_METHODS_ANN_LAYER_FLEXIBLE_RELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FLEXIBLE_RELU_IMPL_HPP

#include "flexible_relu.hpp"

namespace mlpack {

template<typename MatType>
FlexibleReLUType<MatType>::FlexibleReLUType(const double userAlpha) :
    Layer<MatType>(),
    userAlpha(userAlpha)
{
  // Nothing to do here.
}

template<typename MatType>
FlexibleReLUType<MatType>::FlexibleReLUType(
    const FlexibleReLUType& other) :
    Layer<MatType>(other),
    userAlpha(other.userAlpha)
{
  // Nothing to do here.
}

template<typename MatType>
FlexibleReLUType<MatType>::FlexibleReLUType(
    FlexibleReLUType&& other) :
    Layer<MatType>(std::move(other)),
    userAlpha(std::move(other.userAlpha))
{
  // Nothing to do here.
}

template<typename MatType>
FlexibleReLUType<MatType>&
FlexibleReLUType<MatType>::operator=(const FlexibleReLUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    userAlpha = other.userAlpha;
  }

  return *this;
}

template<typename MatType>
FlexibleReLUType<MatType>&
FlexibleReLUType<MatType>::operator=(FlexibleReLUType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    userAlpha = std::move(other.userAlpha);
  }

  return *this;
}

template<typename MatType>
void FlexibleReLUType<MatType>::SetWeights(const MatType& weights)
{
  MakeAlias(alpha, weights, 1, 1);
}

template<typename MatType>
void FlexibleReLUType<MatType>::CustomInitialize(
    MatType& W,
    const size_t elements)
{
  if (elements != 1)
  {
    throw std::invalid_argument("FlexibleReLUType::CustomInitialize(): wrong "
        "elements size!");
  }

  W(0) = userAlpha;
}

template<typename MatType>
void FlexibleReLUType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = arma::clamp(input, 0.0,
      std::numeric_limits<typename MatType::elem_type>::max()) + alpha(0);
}

template<typename MatType>
void FlexibleReLUType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // Compute the first derivative of FlexibleReLU function.
  g = gy % (input > 0);
}

template<typename MatType>
void FlexibleReLUType<MatType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  gradient(0) = accu(error) / input.n_cols;
}

template<typename MatType>
template<typename Archive>
void FlexibleReLUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version*/)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
