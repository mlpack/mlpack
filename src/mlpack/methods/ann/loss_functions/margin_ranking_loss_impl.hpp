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

template<typename MatType>
MarginRankingLossType<MatType>::MarginRankingLossType(
    const double margin, const bool reduction) :
    margin(margin),
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type MarginRankingLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  const int predictionRows = prediction.n_rows;
  const MatType& prediction1 = prediction.rows(0,
      predictionRows / 2 - 1);
  const MatType& prediction2 = prediction.rows(predictionRows / 2,
      predictionRows - 1);

  double lossSum = accu(max(zeros(size(target)),
      -target % (prediction1 - prediction2) + margin));

  if (reduction)
    return lossSum;

  return lossSum / target.n_elem;
}

template<typename MatType>
void MarginRankingLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  const int predictionRows = prediction.n_rows;
  const MatType& prediction1 = prediction.rows(0,
      predictionRows / 2 - 1);
  const MatType& prediction2 = prediction.rows(predictionRows / 2,
      predictionRows - 1);
  MatType lossPrediction1 = -target % (prediction1 - prediction2) + margin;
  lossPrediction1.elem(arma::find(lossPrediction1 >= 0)).ones();
  lossPrediction1.elem(arma::find(lossPrediction1 < 0)).zeros();
  MatType lossPrediction2 = lossPrediction1;
  lossPrediction1 = -target % lossPrediction1;
  lossPrediction2 = target % lossPrediction2;
  loss = join_cols(lossPrediction1, lossPrediction2);

  if (!reduction)
    loss = loss / target.n_elem;
}

template<typename MatType>
template<typename Archive>
void MarginRankingLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(margin));
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
