/**
 * @file methods/ann/layer/scale_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Scale layer, which multiplies its inputs by a constant
 * value.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SCALE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SCALE_IMPL_HPP

// In case it hasn't yet been included.
#include "scale.hpp"

namespace mlpack {

template<typename MatType>
Scale<MatType>::Scale(const typename MatType::elem_type scaleFactor) :
    Layer<MatType>(),
    scaleFactor(scaleFactor)
{
  // Nothing to do.
}

template<typename MatType>
Scale<MatType>::Scale(const Scale& other) :
    Layer<MatType>(other),
    scaleFactor(other.scaleFactor)
{
  // Nothing to do.
}

template<typename MatType>
Scale<MatType>::Scale(
    Scale&& other) :
    Layer<MatType>(std::move(other)),
    scaleFactor(std::move(other.scaleFactor))
{
  // Nothing to do.
}

template<typename MatType>
Scale<MatType>&
Scale<MatType>::operator=(const Scale& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    scaleFactor = other.scaleFactor;
  }

  return *this;
}

template<typename MatType>
Scale<MatType>&
Scale<MatType>::operator=(Scale&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    scaleFactor = std::move(other.scaleFactor);
  }

  return *this;
}

template<typename MatType>
void Scale<MatType>::Forward(
    const MatType& input, MatType& output)
{
  output = input * scaleFactor;
}

template<typename MatType>
void Scale<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy * scaleFactor;
}

template<typename MatType>
template<typename Archive>
void Scale<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(scaleFactor));
}

} // namespace mlpack

#endif
