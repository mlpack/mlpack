/**
 * @file methods/ann/loss_functions/margin_ranking_loss_impl.hpp
 * @author Andrei Mihalea
 *
 * Implementation of the Margin Ranking Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MARGIN_IMPL_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MARGIN_IMPL_LOSS_HPP

// In case it hasn't been included.
#include "margin_ranking_loss.hpp"

namespace mlpack {
namespace ann /** Artifical Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MarginRankingLoss<InputDataType, OutputDataType>::MarginRankingLoss(
    const double margin, const bool reduction):
    margin(margin), reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename PredictionType, typename TargetType>
typename PredictionType::elem_type
MarginRankingLoss<InputDataType, OutputDataType>::Forward(
    const PredictionType& prediction,
    const TargetType& target)
{
  const int predictionRows = prediction.n_rows;
  const PredictionType& prediction1 = prediction.rows(0,
      predictionRows / 2 - 1);
  const PredictionType& prediction2 = prediction.rows(predictionRows / 2,
      predictionRows - 1);

  double lossSum = arma::accu(arma::max(arma::zeros(size(target)),
      -target % (prediction1 - prediction2) + margin));

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template <
    typename PredictionType,
    typename TargetType,
    typename LossType
>
void MarginRankingLoss<InputDataType, OutputDataType>::Backward(
    const PredictionType& prediction,
    const TargetType& target,
    LossType& loss)
{
  const int predictionRows = prediction.n_rows;
  const PredictionType& prediction1 = prediction.rows(0,
      predictionRows / 2 - 1);
  const PredictionType& prediction2 = prediction.rows(predictionRows / 2,
      predictionRows - 1);
  LossType lossPrediction1 = -target % (prediction1 - prediction2) + margin;
  lossPrediction1.elem(arma::find(lossPrediction1 >= 0)).ones();
  lossPrediction1.elem(arma::find(lossPrediction1 < 0)).zeros();
  LossType lossPrediction2 = lossPrediction1;
  lossPrediction1 = -target % lossPrediction1;
  lossPrediction2 = target % lossPrediction2;
  loss = arma::join_cols(lossPrediction1, lossPrediction2);

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MarginRankingLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(margin));
  ar(CEREAL_NVP(reduction));
}

} // namespace ann
} // namespace mlpack

#endif
