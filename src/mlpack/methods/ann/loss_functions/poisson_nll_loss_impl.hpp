/**
 * @file methods/ann/loss_functions/poisson_nll_loss_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the PoissonNLLLossType class.
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

#include <mlpack/core/util/log.hpp>

namespace mlpack {

template<typename MatType>
PoissonNLLLossType<MatType>::PoissonNLLLossType(
    const bool logInput,
    const bool full,
    const typename MatType::elem_type eps,
    const bool reduction) :
    logInput(logInput),
    full(full),
    eps(eps),
    reduction(reduction)
{
  Log::Assert(eps >= 0, "Epsilon (eps) must be greater than or equal to zero.");
}

template<typename MatType>
typename MatType::elem_type PoissonNLLLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  MatType loss(arma::size(prediction));

  if (logInput)
    loss = exp(prediction) - target % prediction;
  else
  {
    CheckProbs(prediction);
    loss = prediction - target % log(prediction + eps);
  }

  if (full)
  {
    const auto mask = target > 1.0;
    const MatType approx = target % log(target) - target
        + 0.5 * log(2 * M_PI * target);
    loss.elem(arma::find(mask)) += approx.elem(arma::find(mask));
  }
  typename MatType::elem_type lossSum = accu(loss);

  if (reduction)
    return lossSum;

  return lossSum / loss.n_elem;
}

template<typename MatType>
void PoissonNLLLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  loss.set_size(size(prediction));

  if (logInput)
    loss = (exp(prediction) - target);
  else
    loss = (1 - target / (prediction + eps));

  if (!reduction)
    loss = loss / loss.n_elem;
}

template<typename MatType>
template<typename Archive>
void PoissonNLLLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(logInput));
  ar(CEREAL_NVP(full));
  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
