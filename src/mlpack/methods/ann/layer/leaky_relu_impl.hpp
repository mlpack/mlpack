/**
 * @file methods/ann/layer/leaky_relu_impl.hpp
 * @author Dhawal Arora
 *
 * Implementation of LeakyReLU layer first introduced in the acoustic model,
 * Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng,
 * "Rectifier Nonlinearities Improve Neural Network Acoustic Models", 2014
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LEAKYRELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LEAKYRELU_IMPL_HPP

// In case it hasn't yet been included.
#include "leaky_relu.hpp"

namespace mlpack {

template<typename MatType>
LeakyReLUType<MatType>::LeakyReLUType(const typename MatType::elem_type alpha) :
    Layer<MatType>(),
    alpha(alpha)
{
  // Nothing to do here.
}

template<typename MatType>
LeakyReLUType<MatType>::LeakyReLUType(const LeakyReLUType& other) :
    Layer<MatType>(other),
    alpha(other.alpha)
{
  // Nothing to do.
}

template<typename MatType>
LeakyReLUType<MatType>::LeakyReLUType(
    LeakyReLUType&& other) :
    Layer<MatType>(std::move(other)),
    alpha(std::move(other.alpha))
{
  // Nothing to do.
}

template<typename MatType>
LeakyReLUType<MatType>&
LeakyReLUType<MatType>::operator=(const LeakyReLUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    alpha = other.alpha;
  }

  return *this;
}

template<typename MatType>
LeakyReLUType<MatType>&
LeakyReLUType<MatType>::operator=(LeakyReLUType&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    alpha = std::move(other.alpha);
  }

  return *this;
}

template<typename MatType>
void LeakyReLUType<MatType>::Forward(const MatType& input, MatType& output)
{
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_elem; ++i)
    output(i) = std::max(input(i), (typename MatType::elem_type) alpha *
        input(i));
}

template<typename MatType>
void LeakyReLUType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  #pragma omp for
  for (size_t i = 0; i < (size_t) input.n_elem; ++i)
    g(i) = gy(i) * ((input(i) >= 0) ? 1 : alpha);
}

template<typename MatType>
template<typename Archive>
void LeakyReLUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
