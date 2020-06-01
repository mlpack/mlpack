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
namespace ann /** Artifical Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::
MultiLabelSoftMarginLoss(
    const bool reduction,
    const arma::rowvec& weights) :
    reduction(reduction),
    weighted(false)
{
  if (weights.n_elem)
  {
    classWeights = weights;
    weighted = true;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input, const TargetType& target)
{
  if (!weighted)
  {
    classWeights.ones(1, input.n_cols);
    weighted = true;
  }

  InputType logSigmoid = arma::log((1 / (1 + arma::exp(-input))));
  InputType logSigmoidNeg = arma::log(1 / (1 + arma::exp(input)));
  InputType loss = arma::mean(arma::sum(-(target % logSigmoid +
      (1 - target) % logSigmoidNeg)) % classWeights, 1);

  if (reduction)
    return arma::as_scalar(loss);

  return arma::as_scalar(loss / input.n_rows);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output.set_size(size(input));
  InputType sigmoid = (1 / (1 + arma::exp(-input)));
  output = -(target % (1 - sigmoid) - (1 - target) % sigmoid) %
        arma::repmat(classWeights, target.n_rows, 1) / output.n_elem;

  if (reduction)
    output = output * input.n_rows;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(classWeights);
  ar & BOOST_SERIALIZATION_NVP(reduction);
}

} // namespace ann
} // namespace mlpack

#endif
