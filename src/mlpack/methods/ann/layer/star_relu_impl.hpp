/**
 * @file methods/ann/layer/star_relu_impl.hpp
 * @author Mayank Raj
 *
 * Implementation of the StarReLU clss.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_STAR_RELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_STAR_RELU_IMPL_HPP

// In case it hasn't yet been included.
#include "star_relu.hpp"

namespace mlpack {

template<typename MatType>
StarReLUType<MatType>::StarReLUType(const double s, const double b) :
    Layer<MatType>(),
    s(s),
    b(b)
{
  // Nothing to do here.
}

template<typename MatType>
StarReLUType<MatType>::StarReLUType(const StarReLUType& other) :
    Layer<MatType>(other),
    s(other.s),
    b(other.b)
{
  // Nothing to do.
}

template<typename MatType>
StarReLUType<MatType>::StarReLUType(StarReLUType&& other) :
    Layer<MatType>(std::move(other)),
    s(std::move(other.s)),
    b(std::move(other.b))
{
  // Nothing to do.
}

template<typename MatType>
StarReLUType<MatType>&
StarReLUType<MatType>::operator=(const StarReLUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    s = other.s;
    b = other.b;
  }

  return *this;
}

template<typename MatType>
StarReLUType<MatType>&
StarReLUType<MatType>::operator=(StarReLUType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    s = std::move(other.s);
    b = std::move(other.b);
  }

  return *this;
}

template<typename MatType>
void StarReLUType<MatType>::Forward(const MatType& input, MatType& output)
{
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_elem; ++i)
  {
    if (input(i) >= 0)
      output(i) = s * std::pow(std::max(input(i), 0.0), 2) + b;
    else
      output(i) = b;
  }
}

template<typename MatType>
void StarReLUType<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_elem; ++i)
  {
    if (input(i) >= 0)
      g(i) = gy(i) * 2 * s * input(i);
    else
      g(i) = 0;
  }
}

template<typename MatType>
template<typename Archive>
void StarReLUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(s));
  ar(CEREAL_NVP(b));
}

} // namespace mlpack

#endif