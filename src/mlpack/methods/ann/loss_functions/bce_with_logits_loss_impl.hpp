/**
 * @file bce_with_logits_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the binary cross-entropy with logits loss function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_BCE_WITH_LOGITS_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_BCE_WITH_LOGITS_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "bce_with_logits_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
BCEWithLogitsLoss<InputDataType, OutputDataType>::BCEWithLogitsLoss(
    const double weight,
    const double posWeight,
    const bool reduction) :
    weight(weight),
    posWeight(posWeight),
    reduction(reduction)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double BCEWithLogitsLoss<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  double lossThis;
  double totalLossReduced = 0;
  double m = input.max();
  double logSigmoidX;
  double logOneMinusSigmoidX;

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    logSigmoidX = m - std::log(std::exp(m) + std::exp(m - input[i]));
    logOneMinusSigmoidX = m - std::log(std::exp(m) + std::exp(m + input[i]));

    lossThis =  - weight * (posWeight * target[i] * logSigmoidX
        + (1 - target[i]) * logOneMinusSigmoidX);

    totalLossReduced += lossThis;
  }
  return reduction ? totalLossReduced / input.n_elem : totalLossReduced;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void BCEWithLogitsLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& input,
    const TargetType&& target,
    OutputType&& output)
{
  output.set_size(size(input));
  double sigmoidX;

  for (size_t i = 0; i < output.n_elem; ++i)
  {
    sigmoidX = 1 / (1 + std::exp(-input[i]));

    output[i] = - weight * (posWeight * target[i] * (1 - sigmoidX)
      - (1 - target[i]) * sigmoidX) / output.n_elem;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BCEWithLogitsLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
