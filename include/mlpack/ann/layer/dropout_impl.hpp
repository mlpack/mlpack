/**
 * @file methods/ann/layer/dropout_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Dropout class, which implements a regularizer that
 * randomly sets units to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPOUT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPOUT_IMPL_HPP

// In case it hasn't yet been included.
#include "dropout.hpp"

namespace mlpack {

template<typename MatType>
DropoutType<MatType>::DropoutType(
    const double ratio) :
    Layer<MatType>(),
    ratio(ratio),
    scale(1.0 / (1.0 - ratio))
{
  // Nothing to do here.
}

template<typename MatType>
DropoutType<MatType>::DropoutType(const DropoutType& other) :
    Layer<MatType>(other),
    ratio(other.ratio),
    scale(other.scale)
{
  // Nothing to do.
}

template<typename MatType>
DropoutType<MatType>::DropoutType(DropoutType&& other) :
    Layer<MatType>(std::move(other)),
    ratio(std::move(other.ratio)),
    scale(std::move(other.scale))
{
  // Nothing to do.
}

template<typename MatType>
DropoutType<MatType>&
DropoutType<MatType>::operator=(const DropoutType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    ratio = other.ratio;
    scale = other.scale;
  }

  return *this;
}

template<typename MatType>
DropoutType<MatType>&
DropoutType<MatType>::operator=(DropoutType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    ratio = std::move(other.ratio);
    scale = std::move(other.scale);
  }

  return *this;
}

template<typename MatType>
void DropoutType<MatType>::Forward(const MatType& input, MatType& output)
{
  // The dropout mask will not be multiplied in testing mode.
  if (!this->training)
  {
    output = input;
  }
  else
  {
    // Scale with input / (1 - ratio) and set values to zero with probability
    // 'ratio'.
    mask.randu(input.n_rows, input.n_cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < input.n_rows; ++i)
    {
      for (size_t j = 0; j < input.n_cols; ++j)
      {
        mask(i, j) = (mask(i, j) > this->ratio) ? 1.0 : 0.0;
      }
    }
    output = input % mask * this->scale;
  }
}

template<typename MatType>
void DropoutType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy % mask * scale;
}

template<typename MatType>
template<typename Archive>
void DropoutType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(ratio));

  // Reset scale.
  scale = 1.0 / (1.0 - ratio);
}

} // namespace mlpack

#endif
