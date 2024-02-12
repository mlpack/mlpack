/**
 * @file methods/ann/loss_functions/dice_loss_impl.hpp
 * @author N Rajiv Vaidyanathan
 *
 * Implementation of the dice loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_DICE_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "dice_loss.hpp"

namespace mlpack {

template<typename MatType>
DiceLossType<MatType>::DiceLossType(const double smooth) : smooth(smooth)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type DiceLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  return 1 - ((2 * accu(target % prediction) + smooth) /
      (accu(target % target) + accu(
      prediction % prediction) + smooth));
}

template<typename MatType>
void DiceLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss = -2 * (target * (accu(prediction % prediction) +
      accu(target % target) + smooth) - prediction *
      (2 * accu(target % prediction) + smooth)) / std::pow(
      accu(target % target) + accu(prediction % prediction)
      + smooth, 2.0);
}

template<typename MatType>
template<typename Archive>
void DiceLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(smooth));
}

} // namespace mlpack

#endif
