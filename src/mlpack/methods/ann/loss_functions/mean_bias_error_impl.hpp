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

template<typename InputDataType, typename OutputDataType>
MeanBiasError<InputDataType, OutputDataType>::MeanBiasError()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
MeanBiasError<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  return arma::accu(target - prediction) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void MeanBiasError<InputDataType, OutputDataType>::Backward(
    const PredictionType& prediction,
    const TargetType& /* target */,
    LossType& loss)
{
  loss.set_size(arma::size(prediction));
  loss.fill(-1.0);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MeanBiasError<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
