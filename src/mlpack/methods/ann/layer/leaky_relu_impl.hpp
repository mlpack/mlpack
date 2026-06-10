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
LeakyReLU<MatType>::LeakyReLU(const typename MatType::elem_type alpha) :
    Layer<MatType>(),
    alpha(alpha)
{
  Log::Assert(0 < alpha && alpha < 1, "Alpha must be in the range 0 to 1");
}

template<typename MatType>
LeakyReLU<MatType>::LeakyReLU(const LeakyReLU& other) :
    Layer<MatType>(other),
    alpha(other.alpha)
{
  Log::Assert(0 < alpha && alpha < 1, "Alpha must be in the range 0 to 1");
}

template<typename MatType>
LeakyReLU<MatType>::LeakyReLU(
    LeakyReLU&& other) :
    Layer<MatType>(std::move(other)),
    alpha(std::move(other.alpha))
{
  Log::Assert(0 < alpha && alpha < 1, "Alpha must be in the range 0 to 1");
}

template<typename MatType>
LeakyReLU<MatType>&
LeakyReLU<MatType>::operator=(const LeakyReLU& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    alpha = other.alpha;
    Log::Assert(0 < alpha && alpha < 1, "Alpha must be in the range 0 to 1");
  }

  return *this;
}

template<typename MatType>
LeakyReLU<MatType>&
LeakyReLU<MatType>::operator=(LeakyReLU&& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(std::move(other));
    alpha = std::move(other.alpha);
    Log::Assert(0 < alpha && alpha < 1, "Alpha must be in the range 0 to 1");
  }

  return *this;
}

template<typename MatType>
void LeakyReLU<MatType>::Forward(const MatType& input, MatType& output)
{
  if constexpr (IsArma<MatType>::value)
  {
    // Empirical simulations shows that the benefit of parallelization is
    // greater than the overhead of spawning OpenMP threads at around 3500
    // elements. The number 3504 was chosen because it is evenly divisible by
    // common core counts (4, 6, 8, 12, 16).
    if (input.n_elem >= 3504)
    {
      #pragma omp parallel for
      for (size_t i = 0; i < (size_t) input.n_elem; ++i) {
        const typename MatType::elem_type input_i = input[i];
        output[i] = std::max(input_i, alpha * input_i);
      }
    }
    else
    {
      for (size_t i = 0; i < (size_t) input.n_elem; ++i) {
        const typename MatType::elem_type input_i = input[i];
        output[i] = std::max(input_i, alpha * input_i);
      }
    }
  }
  else
  {
    output = max(input, alpha * input);
  }
}

template<typename MatType>
void LeakyReLU<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  if constexpr (IsArma<MatType>::value)
  {
    if (input.n_elem >= 3504)
    {
      #pragma omp parallel for
      for (size_t i = 0; i < (size_t) input.n_elem; ++i)
        g[i] = gy[i] * ((input[i] >= 0) ? 1 : alpha);
    }
    else
    {
      for (size_t i = 0; i < (size_t) input.n_elem; ++i)
        g[i] = gy[i] * ((input[i] >= 0) ? 1 : alpha);
    }
  }
  else
  {
    g = gy;
    g.elem(find(input < 0)) *= alpha;
  }
}

template<typename MatType>
template<typename Archive>
void LeakyReLU<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif
