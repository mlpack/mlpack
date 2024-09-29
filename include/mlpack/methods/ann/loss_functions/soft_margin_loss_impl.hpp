/**
 * @file methods/ann/loss_functions/soft_margin_loss_impl.hpp
 * @author Anjishnu Mukherjee
 *
 * Implementation of the Soft Margin Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_IMPL_HPP

// In case it hasn't been included.
#include "soft_margin_loss.hpp"

namespace mlpack {

template<typename MatType>
SoftMarginLossType<MatType>::
SoftMarginLossType(const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type SoftMarginLossType<MatType>::Forward(
    const MatType& prediction, const MatType& target)
{
  MatType loss = log(1 + exp(-target % prediction));
  typename MatType::elem_type lossSum = accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / prediction.n_elem;
}

template<typename MatType>
void SoftMarginLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss.set_size(size(prediction));
  MatType temp = exp(-target % prediction);
  MatType numerator = -target % temp;
  MatType denominator = 1 + temp;
  loss = numerator / denominator;

  if (!reduction)
    loss = loss / prediction.n_elem;
}

template<typename MatType>
template<typename Archive>
void SoftMarginLossType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
