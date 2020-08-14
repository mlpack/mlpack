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
    const double margin) : margin(margin)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MarginRankingLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  const int inputRows = input.n_rows;
  const InputType& input1 = input.rows(0, inputRows / 2 - 1);
  const InputType& input2 = input.rows(inputRows / 2, inputRows - 1);
  return arma::accu(arma::max(arma::zeros(size(target)),
      -target % (input1 - input2) + margin)) / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template <
    typename InputType,
    typename TargetType,
    typename OutputType
>
void MarginRankingLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  const int inputRows = input.n_rows;
  const InputType& input1 = input.rows(0, inputRows / 2 - 1);
  const InputType& input2 = input.rows(inputRows / 2, inputRows - 1);
  output = -target % (input1 - input2) + margin;
  output.elem(arma::find(output >= 0)).ones();
  output.elem(arma::find(output < 0)).zeros();
  output = (input2 - input1) % output / target.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MarginRankingLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(margin);
}

} // namespace ann
} // namespace mlpack

#endif
