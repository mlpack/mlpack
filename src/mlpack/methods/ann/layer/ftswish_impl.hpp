/**
 * @file methods/ann/layer/ftswish_impl.hpp
 * @author Mayank Raj
 *
 * Definition of Flatten T Swish layer first introduced in the acoustic model,
 * Hock Hung Chieng, Noorhaniza Wahid, Pauline Ong, Sai Raj Kishore Perla,
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
FTSwish<MatType>::FTSwish(const double T) :
    Layer<MatType>(),
    T(T)
{
  // Nothing to do here.
}

template<typename MatType>
FTSwish<MatType>::FTSwish(const FTSwish& other) :
    Layer<MatType>(other),
    T(other.T)
{
  // Nothing to do.
}

template<typename MatType>
FTSwish<MatType>::FTSwish(FTSwish&& other) :
    Layer<MatType>(std::move(other)),
    T(std::move(other.T))
{
  // Nothing to do.
}

template<typename MatType>
FTSwish<MatType>&
FTSwish<MatType>::operator=(const FTSwish& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    T = other.T;
  }

  return *this;
}

template<typename MatType>
FTSwish<MatType>&
FTSwish<MatType>::operator=(FTSwish&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    T = std::move(other.T);
  }

  return *this;
}

template<typename MatType>
void FTSwish<MatType>::Forward(const MatType& input, MatType& output)
{
  output = ElemType(T) + clamp(input, 0,
      std::numeric_limits<ElemType>::max()) /
      (1 + exp(-input));
}

template<typename MatType>
void FTSwish<MatType>::Backward(
    const MatType& input,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  const ElemType convT = ElemType(T);
  g = gy % sign(output - convT) % (((output - convT) / input) %
      (1 - (output - convT)) + (output - convT));
}

template<typename MatType>
template<typename Archive>
void FTSwish<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(T));
}

} // namespace mlpack

#endif
