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
    const double weight,
    const bool reduction) :
    weight(weight),
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input, const TargetType& target)
{
  InputType logSigmoid = 1 / (1 + arma::exp(-input));
  InputType logSigmoidNeg = 1 / (1 + arma::exp(input));
  double loss = arma::accu(-(target % logSigmoid +
      (1 - target) % logSigmoidNeg) * weight);
  return reduction ? loss / input.n_elem : loss / input.n_rows;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void MultiLabelSoftMarginLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  InputType expo = arma::exp(input);
  output = (expo / arma::pow(1 + expo, 2)) %
      ((target + 1) / arma::log(1 + expo) -
      target / (input - arma::log(expo + 1))) * weight;
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
