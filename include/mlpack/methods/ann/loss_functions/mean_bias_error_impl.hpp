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

template<typename MatType>
MeanBiasErrorType<MatType>::MeanBiasErrorType(const bool reduction) :
    reduction(reduction)
{
  // Nothing to do here
}

template<typename MatType>
typename MatType::elem_type MeanBiasErrorType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType loss = target - prediction;
  typename MatType::elem_type lossSum = accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void MeanBiasErrorType<MatType>::Backward(
    const MatType& prediction,
    const MatType& /* target */,
    MatType& loss)
{
  loss.set_size(arma::size(prediction));
  loss.fill(-1.0);

  if (!reduction)
    loss = loss / loss.n_elem;
}

template<typename MatType>
template<typename Archive>
void MeanBiasErrorType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
