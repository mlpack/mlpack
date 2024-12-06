/**
 * @file methods/ann/loss_functions/multilabel_softmargin_loss_impl.hpp
 * @author Anjishnu Mukherjee
 *
 * Implementation of the Multi Label Soft Margin Loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_MULTILABEL_SOFTMARGIN_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_MULTILABEL_SOFTMARGIN_LOSS_IMPL_HPP

// In case it hasn't been included.
#include "multilabel_softmargin_loss.hpp"

namespace mlpack {

template<typename MatType>
MultiLabelSoftMarginLossType<MatType>::MultiLabelSoftMarginLossType(
    const bool reduction,
    const arma::Row<typename MatType::elem_type>& weights) :
    reduction(reduction),
    weighted(false)
{
  if (weights.n_elem)
  {
    classWeights = weights;
    weighted = true;
  }
}

template<typename MatType>
typename MatType::elem_type MultiLabelSoftMarginLossType<MatType>::Forward(
    const MatType& input, const MatType& target)
{
  if (!weighted)
  {
    classWeights.ones(1, input.n_cols);
    weighted = true;
  }

  MatType logSigmoid = log((1 / (1 + exp(-input))));
  MatType logSigmoidNeg = log(1 / (1 + exp(input)));
  MatType loss = arma::mean(sum(-(target % logSigmoid +
      (1 - target) % logSigmoidNeg)) % classWeights, 1);

  if (reduction)
    return arma::as_scalar(loss);

  return arma::as_scalar(loss / input.n_rows);
}

template<typename MatType>
void MultiLabelSoftMarginLossType<MatType>::Backward(
    const MatType& input,
    const MatType& target,
    MatType& output)
{
  output.set_size(size(input));
  MatType sigmoid = (1 / (1 + exp(-input)));
  output = -(target % (1 - sigmoid) - (1 - target) % sigmoid) %
        repmat(classWeights, target.n_rows, 1) / output.n_elem;

  if (reduction)
    output = output * input.n_rows;
}

template<typename MatType>
template<typename Archive>
void MultiLabelSoftMarginLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(classWeights));
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
