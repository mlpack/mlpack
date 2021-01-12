/**
 * @file methods/ann/loss_functions/soft_margin_loss_impl.hpp
 * @author Anjishnu Mukherjee
 *
 * Implementation of the Soft Margin Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_SOFT_MARGIN_LOSS_IMPL_HPP

// In case it hasn't been included.
#include "soft_margin_loss.hpp"

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
SoftMarginLoss<InputDataType, OutputDataType>::
SoftMarginLoss(const bool reduction) : reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
SoftMarginLoss<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction, const TargetType& target)
{
  PredictionType loss = arma::log(1 + arma::exp(-target % prediction));
  typename PredictionType::elem_type lossSum = arma::accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / prediction.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void SoftMarginLoss<InputDataType, OutputDataType>::Backward(
    const PredictionType& prediction,
    const TargetType& target,
    LossType& loss)
{
  loss.set_size(size(prediction));
  PredictionType temp = arma::exp(-target % prediction);
  PredictionType numerator = -target % temp;
  PredictionType denominator = 1 + temp;
  loss = numerator / denominator;

  if (!reduction)
    loss = loss / prediction.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void SoftMarginLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace ann
} // namespace mlpack

#endif
