/**
 * @file multilabel_softmargin_loss_impl.hpp
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
    arma::mat weight,
    const size_t num_classes,
    const bool reduction) :
    num_classes(num_classes),
    reduction(reduction)
{
  class_weights.ones(1, num_classes);
  class_weights = weight;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input, const TargetType& target)
{
  InputType logSigmoid = arma::log((1 / (1 + arma::exp(-input))));
  InputType logSigmoidNeg = arma::log(1 / (1 + arma::exp(input)));
  InputType loss = arma::mean(arma::mean(-(target % logSigmoid +
      (1 - target) % logSigmoidNeg), 1) * class_weights, 1);

  if (reduction)
  {
    return arma::as_scalar(arma::sum(loss));
  }
  else
  {
    return arma::as_scalar(arma::mean(loss));
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output.set_size(size(x));
  InputType Sigmoid = (1 / (1 + arma::exp(-input)));
  output = - (target % (1-Sigmoid) - (1-target) % Sigmoid) %
      arma::repmat(class_weights, target.n_rows, 1) / output.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
