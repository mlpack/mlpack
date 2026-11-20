/**
 * @file methods/ann/loss_functions/mean_squared_error_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the mean squared error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_SQUARED_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_squared_error.hpp"

namespace mlpack {

template<typename MatType>
MeanSquaredErrorType<MatType>::MeanSquaredErrorType(const bool reduction) :
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type MeanSquaredErrorType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  typename MatType::elem_type lossSum = accu(square(prediction - target));

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void MeanSquaredErrorType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss = 2 * (prediction - target);

  if (!reduction)
    loss = loss / target.n_rows /* ignore batch size */;
}

template<typename MatType>
template<typename Archive>
void MeanSquaredErrorType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
