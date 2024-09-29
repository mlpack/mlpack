/**
 * @file methods/ann/layer/celu_impl.hpp
 * @author Gaurav Singh
 *
 * Implementation of the CELU activation function as described by Jonathan T. Barron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CELU_IMPL_HPP

// In case it hasn't yet been included.
#include "celu.hpp"

namespace mlpack {

template<typename MatType>
CELUType<MatType>::CELUType(const double alpha) :
    Layer<MatType>(),
    alpha(alpha)
{
  if (alpha == 0)
  {
    Log::Fatal << "The value of alpha cannot be equal to 0, "
               << "terminating the program." << std::endl;
  }
}

template<typename MatType>
CELUType<MatType>::CELUType(const CELUType& other) :
    Layer<MatType>(other),
    alpha(other.alpha)
{
    // Nothing to do.
}

template<typename MatType>
CELUType<MatType>::CELUType(
    CELUType&& other) :
    Layer<MatType>(std::move(other)),
    alpha(std::move(other.alpha))
{
    // Nothing to do.
}

template<typename MatType>
CELUType<MatType>&
CELUType<MatType>::operator=(const CELUType& other)
{
  if (&other != this)
  {
    Layer<MatType>::operator=(other);
    alpha = other.alpha;
  }

  return *this;
}

template<typename MatType>
CELUType<MatType>&
CELUType<MatType>::operator=(CELUType&& other)
{
    if (&other != this)
    {
      Layer<MatType>::operator=(std::move(other));
      alpha = std::move(other.alpha);
    }

    return *this;
}

template<typename MatType>
void CELUType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output(i) = (input(i) >= 0) ? input(i) : alpha *
        (std::exp(input(i) / alpha) - 1);
  }

  if (this->training)
  {
    derivative.set_size(arma::size(input));
    for (size_t i = 0; i < input.n_elem; ++i)
    {
      derivative(i) = (input(i) >= 0) ? 1 :
          (output(i) / alpha) + 1;
    }
  }
}

template<typename MatType>
void CELUType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  g = gy % derivative;
}

template<typename MatType>
template<typename Archive>
void CELUType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(alpha));
  if (Archive::is_loading::value)
    derivative.clear();
}

} // namespace mlpack

#endif
