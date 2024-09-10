/**
 * @file methods/ann/layer/identity_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of the Identity layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_IDENTITY_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_IDENTITY_IMPL_HPP

// In case it hasn't yet been included.
#include "identity.hpp"

namespace mlpack {

template<typename MatType>
IdentityType<MatType>::IdentityType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
IdentityType<MatType>::IdentityType(
    const IdentityType& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
IdentityType<MatType>::IdentityType(
    IdentityType&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
IdentityType<MatType>&
IdentityType<MatType>::operator=(const IdentityType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
  }

  return *this;
}

template<typename MatType>
IdentityType<MatType>&
IdentityType<MatType>::operator=(IdentityType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
  }

  return *this;
}

template<typename MatType>
void IdentityType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = input;
}

template<typename MatType>
void IdentityType<MatType>::Backward(
  const MatType& /* input */,
  const MatType& /* output */,
  const MatType& gy,
  MatType& g)
{
  g = gy;
}

template<typename MatType>
template<typename Archive>
void IdentityType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
}

} // namespace mlpack

#endif
