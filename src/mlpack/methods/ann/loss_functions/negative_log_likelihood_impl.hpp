/**
 * @file methods/ann/loss_functions/negative_log_likelihood_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the NegativeLogLikelihood class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_NEGATIVE_LOG_LIKELIHOOD_IMPL_HPP

// In case it hasn't yet been included.
#include "negative_log_likelihood.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
NegativeLogLikelihood<InputDataType, OutputDataType>::NegativeLogLikelihood()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
NegativeLogLikelihood<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  typedef typename PredictionType::elem_type ElemType;
  ElemType output = 0;
  for (size_t i = 0; i < prediction.n_cols; ++i)
  {
    Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
        "Target class out of range.");

    output -= prediction(target(i), i);
  }

  return output;
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType, typename LossType>
void NegativeLogLikelihood<InputDataType, OutputDataType>::Backward(
      const PredictionType& prediction,
      const TargetType& target,
      LossType& loss)
{
  loss = arma::zeros<LossType>(prediction.n_rows, prediction.n_cols);
  for (size_t i = 0; i < prediction.n_cols; ++i)
  {
    Log::Assert(target(i) >= 0 && target(i) < prediction.n_rows,
        "Target class out of range.");

    loss(target(i), i) = -1;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void NegativeLogLikelihood<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */, const uint32_t /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
