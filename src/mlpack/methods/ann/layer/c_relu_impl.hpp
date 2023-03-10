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

template<typename MatType>
CReLUType<MatType>::CReLUType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
CReLUType<MatType>::CReLUType(
    const CReLUType& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
CReLUType<MatType>::CReLUType(
    CReLUType&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
CReLUType<MatType>&
CReLUType<MatType>::operator=(const CReLUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
CReLUType<MatType>&
CReLUType<MatType>::operator=(CReLUType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
  }

  return *this;
}

template<typename MatType>
void CReLUType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = arma::join_cols(arma::max(input, 0.0 * input), arma::max(
      (-1 * input), 0.0 * input));
}

template<typename MatType>
void CReLUType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  MatType temp = gy % (input >= 0.0);
  g = temp.rows(0, (input.n_rows / 2 - 1)) - temp.rows(input.n_rows / 2,
      (input.n_rows - 1));
}

template<typename MatType>
template<typename Archive>
void CReLUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
}

} // namespace mlpack

#endif
