/**
 * @file methods/ann/loss_functions/hinge_loss_impl.hpp
 * @author Anush Kini
 *
 * Implementation of the Hinge loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_HINGE_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "hinge_loss.hpp"

namespace mlpack {

template<typename MatType>
HingeLossType<MatType>::HingeLossType(const bool reduction):
  reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type HingeLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType temp = target - (target == 0);
  MatType tempZeros(size(target));

  MatType loss = max(tempZeros, 1 - prediction % temp);

  typename MatType::elem_type lossSum = accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / loss.n_elem;
}

template<typename MatType>
void HingeLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  MatType temp = target - (target == 0);
  loss = (prediction < (1 / temp)) % -temp;

  if (!reduction)
    loss /= target.n_elem;
}

template<typename MatType>
template<typename Archive>
void HingeLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
