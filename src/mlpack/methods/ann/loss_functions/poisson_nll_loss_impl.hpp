/**
 * @file methods/ann/loss_functions/poisson_nll_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the PoissonNLLLoss class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_POISSON_NLL_LOSS_IMPL_HPP


// In case it hasn't yet been included.
#include "poisson_nll_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
PoissonNLLLoss<InputDataType, OutputDataType>::PoissonNLLLoss(
    const bool logInput,
    const bool full,
    const typename InputDataType::elem_type eps,
    const bool mean):
    logInput(logInput),
    full(full),
    eps(eps),
    mean(mean)
{
  Log::Assert(eps >= 0, "Epsilon (eps) must be greater than or equal to zero.");
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputDataType::elem_type
PoissonNLLLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  InputType loss(arma::size(input));

  if (logInput)
    loss = arma::exp(input) - target % input;
  else
  {
    CheckProbs(input);
    loss = input - target % arma::log(input + eps);
  }

  if (full)
  {
    const auto mask = target > 1.0;
    const InputType approx = target % arma::log(target) - target
        + 0.5 * arma::log(2 * M_PI * target);
    loss.elem(arma::find(mask)) += approx.elem(arma::find(mask));
  }

  return mean ? arma::accu(loss) / loss.n_elem : arma::accu(loss);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void PoissonNLLLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output.set_size(size(input));

  if (logInput)
    output = (arma::exp(input) - target);
  else
    output = (1 - target / (input + eps));

  if (mean)
    output = output / output.n_elem;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void PoissonNLLLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(logInput);
  ar & CEREAL_NVP(full);
  ar & CEREAL_NVP(eps);
  ar & CEREAL_NVP(mean);
}

} // namespace ann
} // namespace mlpack

#endif
