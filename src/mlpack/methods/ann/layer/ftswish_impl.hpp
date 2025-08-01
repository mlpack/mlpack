/**
 * @file methods/ann/layer/ftswish_impl.hpp
 * @author Mayank Raj
 *
 * Definition of Flatten T Swish layer first introduced in the acoustic model,
 * Hock Hung Chieng, Noorhaniza Wahid, Pauline Ong, Sai Raj Kishore Perla,
 * 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_FTSWISH_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_FTSWISH_IMPL_HPP

// In case it hasn't yet been included.
#include "ftswish.hpp"

namespace mlpack {

template<typename MatType>
FTSwishType<MatType>::FTSwishType(const double T) :
    Layer<MatType>(),
    T(T)
{
  // Nothing to do here.
}

template<typename MatType>
FTSwishType<MatType>::FTSwishType(const FTSwishType& other) :
    Layer<MatType>(other),
    T(other.T)
{
  // Nothing to do.
}

template<typename MatType>
FTSwishType<MatType>::FTSwishType(FTSwishType&& other) :
    Layer<MatType>(std::move(other)),
    T(std::move(other.T))
{
  // Nothing to do.
}

template<typename MatType>
FTSwishType<MatType>&
FTSwishType<MatType>::operator=(const FTSwishType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    T = other.T;
  }

  return *this;
}

template<typename MatType>
FTSwishType<MatType>&
FTSwishType<MatType>::operator=(FTSwishType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    T = std::move(other.T);
  }

  return *this;
}

template<typename MatType>
void FTSwishType<MatType>::Forward(const MatType& input, MatType& output)
{
  output = T + clamp(input, 0,
      std::numeric_limits<typename MatType::elem_type>::max()) / (1 + exp(-input));
}

template<typename MatType>
void FTSwishType<MatType>::Backward(
    const MatType& input,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  // The sign() makes sure that we only multiply non-zero input elements.
  g = gy % sign(output - T) % (1 - (1 / (1 + exp(-input))) % (output - T));
}

template<typename MatType>
template<typename Archive>
void FTSwishType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(T));
}

} // namespace mlpack

#endif
