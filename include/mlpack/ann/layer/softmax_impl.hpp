/**
 * @file methods/ann/layer/softmax_impl.hpp
 * @author Mrityunjay Tripathi
 * @author Sreenik Seal
 *
 * Implementation of the Softmax class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SOFTMAX_IMPL_HPP

// In case it hasn't yet been included.
#include "softmax.hpp"

namespace mlpack {

template<typename MatType>
SoftmaxType<MatType>::SoftmaxType() :
    Layer<MatType>()
{
  // Nothing to do here.
}

template<typename MatType>
SoftmaxType<MatType>::SoftmaxType(const SoftmaxType& other) :
    Layer<MatType>(other)
{
  // Nothing to do here.
}

template<typename MatType>
SoftmaxType<MatType>::SoftmaxType(SoftmaxType&& other) :
    Layer<MatType>(std::move(other))
{
  // Nothing to do here.
}

template<typename MatType>
SoftmaxType<MatType>&
SoftmaxType<MatType>::operator=(const SoftmaxType& other)
{
  if (this != &other)
    Layer<MatType>::operator=(other);

  return *this;
}

template<typename MatType>
SoftmaxType<MatType>&
SoftmaxType<MatType>::operator=(SoftmaxType&& other)
{
  if (this != &other)
    Layer<MatType>::operator=(std::move(other));

  return *this;
}

template<typename MatType>
void SoftmaxType<MatType>::Forward(const MatType& input, MatType& output)
{
  MatType softmaxInput = exp(input.each_row() - max(input, 0));
  output = softmaxInput.each_row() / sum(softmaxInput, 0);
}

template<typename MatType>
void SoftmaxType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& output,
    const MatType& gy,
    MatType& g)
{
  g = output % (gy - repmat(sum(gy % output), output.n_rows, 1));
}

template<typename MatType>
template<typename Archive>
void SoftmaxType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
}

} // namespace mlpack

#endif
