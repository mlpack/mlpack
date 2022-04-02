/**
 * @file methods/ann/loss_functions/mean_bias_error_impl.hpp
 * @author Saksham Rastogi
 *
 * Implementation of the mean bias error performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_BIAS_ERROR_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MEAN_BIAS_ERROR_IMPL_HPP


// In case it hasn't yet been included.
#include "mean_bias_error.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename MatType>
MeanBiasErrorType<MatType>::MeanBiasErrorType()
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type MeanBiasErrorType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  return arma::accu(target - prediction) / target.n_cols;
}

template<typename MatType>
void MeanBiasErrorType<MatType>::Backward(
    const MatType& prediction,
    const MatType& /* target */,
    MatType& loss)
{
  loss.set_size(arma::size(prediction));
  loss.fill(-1.0);
}

} // namespace ann
} // namespace mlpack

#endif
