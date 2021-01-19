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
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
HingeLoss<InputDataType, OutputDataType>::HingeLoss(const bool reduction):
  reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
HingeLoss<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  TargetType temp = target - (target == 0);
  TargetType temp_zeros(size(target), arma::fill::zeros);

  PredictionType loss = arma::max(1 - prediction % temp, temp_zeros);

  typename PredictionType::elem_type lossSum = arma::accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / loss.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void HingeLoss<InputDataType, OutputDataType>::Backward(
    const PredictionType& prediction,
    const TargetType& target,
    LossType& loss)
{
  TargetType temp = target - (target == 0);
  loss = (prediction < (1 / temp)) % -temp;

  if (!reduction)
    loss /= target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void HingeLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(reduction));
}

} // namespace ann
} // namespace mlpack

#endif
