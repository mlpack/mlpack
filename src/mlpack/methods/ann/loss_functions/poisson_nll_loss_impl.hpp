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
  InputType loss(size(input), arma::fill::zeros);
  if (logInput)
  {
    loss = arma::exp(input) - target % input;
  }
  else
  {
    loss = input - target % arma::log(input + eps);
  }

  if (full)
  {
    // Stirling's approximation :
    // log(n!) = n * log(n) - n + log(2 * pi * n)
    // Select all elements greater than 1 in target and
    // add ```target * log(target) - target + log(2 * pi * target)```
    loss.elem(find(target > 1)) += target * arma::log(target) - target
        + arma::log(2 * arma::datum::pi * target) / 2;
  }

  if (reduce)
  {
    return arma::accu(loss) / loss.n_elem;
  }
  return arma::accu(loss);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void PoissonNLLLoss<InputDataType, OutputDataType>::Backward(
  const InputType&& input,
  const TargetType&& target,
  OutputType&& output)
{
  if (logInput)
  {
    output = (arma::exp(input) - target) / input.n_elem;
  }
  else
  {
    output = (1 - target / (input + eps)) / input.n_elem;
  }

  if (full)
  {
    // Approximation for (1 + 1/2 + 1/3 + ... + 1/n) is
    // (log(n + 1) + log(n) + 1)/2
    // Select all elements greater than 1 in target and
    // add ```(log(n + 1) + log(n) + 1)/2```
    output.elem(find(target > 1)) += (arma::log(target + 1)
        + arma::log(target) + 1) / (2 * input.n_elem);
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
