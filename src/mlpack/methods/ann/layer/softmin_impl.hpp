/**
 * @file methods/ann/layer/softmin_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Softmin class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMIN_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMIN_IMPL_HPP

// In case it hasn't yet been included.
#include "softmin.hpp"

namespace mlpack {

template<typename MatType>
Softmin<MatType>::Softmin()
{
  // Nothing to do here.
}

template<typename MatType>
Softmin<MatType>::Softmin(const Softmin& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
Softmin<MatType>::Softmin(Softmin&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
Softmin<MatType>&
Softmin<MatType>::operator=(const Softmin& other)
{
  if (this != &other)
    Layer<MatType>::operator=(other);

  return *this;
}

template<typename MatType>
Softmin<MatType>&
Softmin<MatType>::operator=(Softmin&& other)
{
  if (this != &other)
    Layer<MatType>::operator=(std::move(other));

  return *this;
}

template<typename MatType>
void Softmin<MatType>::Forward(
    const MatType& input,
    MatType& output)
{
  MatType softminInput = exp(-(input.each_row() -
      min(input, 0)));
  output = softminInput.each_row() / sum(softminInput, 0);
}

template<typename MatType>
void Softmin<MatType>::Backward(
    const MatType& /* input */,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  g = output % (gy - repmat(sum(gy % output), output.n_rows, 1));
}

template<typename MatType>
template<typename Archive>
void Softmin<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
}

} // namespace mlpack

#endif
