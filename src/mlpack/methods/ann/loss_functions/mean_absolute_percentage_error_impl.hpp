/**
 * @file methods/ann/loss_functions/mean_absolute_percentage_error_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of the Mean Absolute Percentage Error function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_ABSOLUTE_PERCENTAGE_ERROR_IMPL_HPP

// In case it hasn't yet been included.
#include "mean_absolute_percentage_error.hpp"

namespace mlpack {

template<typename MatType>
MeanAbsolutePercentageErrorType<MatType>::MeanAbsolutePercentageErrorType()
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type MeanAbsolutePercentageErrorType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType loss = arma::abs((prediction - target) / target);
  return accu(loss) * (100 / target.n_cols);
}

template<typename MatType>
void MeanAbsolutePercentageErrorType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)

{
  loss = (((ConvTo<arma::mat>::From(prediction < target) * -2) + 1) /
      target) * (100 / target.n_cols);
}

} // namespace mlpack

#endif
