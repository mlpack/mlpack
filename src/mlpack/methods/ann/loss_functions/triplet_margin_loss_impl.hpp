/**
 * @file methods/ann/loss_functions/triplet_margin_loss_impl.hpp
 * @author Prince Gupta
 * @author Ayush Singh
 *
 * Implementation of the Triplet Margin Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_IMPL_LOSS_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_TRIPLET_MARGIN_IMPL_LOSS_HPP

// In case it hasn't been included.
#include "triplet_margin_loss.hpp"

namespace mlpack {

template<typename MatType>
TripletMarginLossType<MatType>::TripletMarginLossType(const double margin) :
    margin(margin)
{
  // Nothing to do here.
}

template<typename MatType>
typename MatType::elem_type TripletMarginLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType anchor =
      prediction.submat(0, 0, prediction.n_rows / 2 - 1, prediction.n_cols - 1);
  MatType positive =
      prediction.submat(prediction.n_rows / 2, 0, prediction.n_rows - 1,
      prediction.n_cols - 1);
  return std::max(0.0, accu(pow(anchor - positive, 2)) -
      accu(pow(anchor - target, 2)) + margin) / anchor.n_cols;
}

template<typename MatType>
void TripletMarginLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  MatType positive =
      prediction.submat(prediction.n_rows / 2, 0, prediction.n_rows - 1,
      prediction.n_cols - 1);
  loss = 2 * (target - positive) / target.n_cols;
}

template<typename MatType>
template<typename Archive>
void TripletMarginLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(margin));
}

} // namespace mlpack

#endif
