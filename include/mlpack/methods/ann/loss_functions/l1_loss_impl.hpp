/**
 * @file methods/ann/loss_functions/l1_loss_impl.hpp
 * @author Himanshu Pathak
 *
 * Implementation of the L1 Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_L1_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "l1_loss.hpp"

namespace mlpack {

template<typename MatType>
L1LossType<MatType>::L1LossType(const bool reduction):
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type L1LossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType loss = arma::abs(prediction - target);
  typename MatType::elem_type lossSum = accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void L1LossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss = sign(prediction - target);

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void L1LossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
