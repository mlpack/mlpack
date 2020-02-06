/**
 * @file poisson_nll_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Poisson NegativeLogLikelihood class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POISSON_NLL_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_POISSON_NLL_LOSS_IMPL_HPP


// In case it hasn't yet been included.
#include "poisson_nll_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PoissonNLLLoss<InputDataType, OutputDataType>::PoissonNLLLoss(
  const bool logInput,
  const bool full,
  const double eps,
  const bool reduce):
  logInput(logInput),
  full(full),
  eps(eps),
  reduce(reduce)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double PoissonNLLLoss<InputDataType, OutputDataType>::Forward(
  const InputType&& input,
  TargetType&& target)
{
  double lossThis;
  double totalLossReduced = 0;

  for (size_t i = 0; i < input.n_elem; ++i)
  {
    if (logInput)
    {
      lossThis = std::exp(input[i]) - target[i] * input[i];
    }
    else
    {
      lossThis = input[i] - target[i] * std::log(input[i] + eps);
    }

    if ((full) && (target[i] > 1))
    {
      lossThis += target[i] * std::log(target[i]) - target[i]
          + std::log(2 * M_PI * target[i]) / 2;
    }

    totalLossReduced += lossThis;
  }

  if (reduce)
  {
    return totalLossReduced / input.n_elem;
  }
  return totalLossReduced;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void PoissonNLLLoss<InputDataType, OutputDataType>::Backward(
  const InputType&& input,
  const TargetType&& target,
  OutputType&& output)
{
  output.set_size(size(input));
  for (size_t i = 0; i < output.n_elem; ++i)
  {
    if (logInput)
    {
      output[i] = (std::exp(input[i]) - target[i]) / input.n_elem;
    }
    else
    {
      output[i] = (1 - target[i] / (input[i] + eps)) / input.n_elem;
    }

    if ((full) && (target[i] > 1))
    {
      // Here, (1 + 1/2 + 1/3 + ... + 1/n) =~ (log(n + 1) + log(n) + 1)/2
      output[i] += (std::log(target[i] + 1)
          + std::log(target[i]) + 1) / (2 * input.n_elem);
    }
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PoissonNLLLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
