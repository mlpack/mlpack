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
namespace ann /** Artificial Neural Network. */ {

template<typename MatType>
L1Loss<MatType>::L1Loss(const bool mean):
  mean(mean)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type L1Loss<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  if (mean)
    return arma::accu(arma::mean(prediction - target));

  return arma::accu(prediction - target);
}

template<typename MatType>
void L1Loss<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss = arma::sign(prediction - target);
}

template<typename MatType>
template<typename Archive>
void L1Loss<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(mean));
}

} // namespace ann
} // namespace mlpack

#endif
