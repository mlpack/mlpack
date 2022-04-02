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
namespace ann /** Artificial Neural Network. */ {

template<typename MatType>
MeanSquaredError<MatType>::MeanSquaredError()
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type MeanSquaredError<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  return arma::accu(arma::square(prediction - target)) / target.n_cols;
}

template<typename MatType>
void MeanSquaredError<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss = 2 * (prediction - target) / target.n_cols;
}

} // namespace ann
} // namespace mlpack

#endif
