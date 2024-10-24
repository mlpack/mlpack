/**
 * @file methods/ann/loss_functions/huber_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Huber loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HUBER_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "huber_loss.hpp"

namespace mlpack {

template<typename MatType>
HuberLossType<MatType>::HuberLossType(
    const double delta,
    const bool reduction):
    delta(delta),
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type HuberLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  using ElemType = typename MatType::elem_type;
  ElemType lossSum = 0;
  for (size_t i = 0; i < prediction.n_elem; ++i)
  {
    const ElemType absError = std::abs(target[i] - prediction[i]);
    lossSum += absError > delta ?
        delta * (absError - 0.5 * delta) : 0.5 * std::pow(absError, 2);
  }

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void HuberLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  using ElemType = typename MatType::elem_type;

  loss.set_size(size(prediction));
  for (size_t i = 0; i < loss.n_elem; ++i)
  {
    const ElemType absError = std::abs(target[i] - prediction[i]);
    loss[i] = absError > delta ?
        -delta * (target[i] - prediction[i]) / absError :
        prediction[i] - target[i];
  }

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void HuberLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(delta));
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
