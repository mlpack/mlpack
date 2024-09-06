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
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_elem; ++i)
  {
    if (input(i) >= 0)
      output(i) = input(i) / (1 + std::exp(-input(i))) + T;
    else
      output(i) = T;
  }
}

template<typename MatType>
void FTSwishType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_elem; ++i)
  {
    if (input(i) >= 0)
    {
      const double sigmoidX = 1 / (1 + std::exp(-input(i)));
      const double fX = input(i) * sigmoidX;

      g(i) = gy(i) * (sigmoidX * (1 - fX) + fX);
    }
    else
    {
      g(i) = 0;
    }
  }
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
