/**
 * @file methods/ann/loss_functions/focal_loss_impl.hpp
 *
 * Implementation of the Focal Loss function with logits.
 *
 * mlpack is free software; you may redistribute it and/or modify it 
 * under the terms of the 3-clause BSD license.  You should have 
 * received a copy of the 3-clause BSD license along with mlpack.  
 * If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTION_FOCAL_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTION_FOCAL_LOSS_IMPL_HPP

#include "focal_loss.hpp"

namespace mlpack {

template<typename MatType>
FocalLossType<MatType>::FocalLossType(
    const double gamma,
    const double alpha,
    const bool reduction) :
    gamma(gamma),
    alpha(alpha),
    reduction(reduction)
{
  // Nothing to do.
}

template<typename MatType>
typename MatType::elem_type FocalLossType<MatType>::Forward(
    const MatType& prediction,
    const MatType& target)
{
  // p = sigmoid(logits)
  MatType p = 1.0 / (1.0 + arma::exp(-prediction));

  // Clamp probabilities to prevent log(0)
  const double eps = 1e-15;
  MatType p_clamped = arma::clamp(p, eps, 1.0 - eps);

  // p_t = p if target == 1 else 1 - p
  MatType p_t = target % p_clamped + (1.0 - target) % (1.0 - p_clamped);

  // alpha_t = alpha if target == 1 else 1 - alpha
  MatType alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha);

  // FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
  MatType loss = -alpha_t % arma::pow(1.0 - p_t, gamma) % arma::log(p_t);

  typename MatType::elem_type lossSum = arma::accu(loss);

  if (!reduction)
    return lossSum / prediction.n_elem;

  return lossSum;
}

template<typename MatType>
void FocalLossType<MatType>::Backward(
    const MatType& prediction,
    const MatType& target,
    MatType& loss)
{
  MatType p = 1.0 / (1.0 + arma::exp(-prediction));

  const double eps = 1e-15;
  MatType p_clamped = arma::clamp(p, eps, 1.0 - eps);

  MatType p_t = target % p_clamped + (1.0 - target) % (1.0 - p_clamped);
  MatType alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha);

  // Gradient w.r.t logits (x):
  // dFL/dx = alpha_t * (1 - p_t)^gamma * (gamma * p_t * log(p_t)
  // + p_t - 1) * (2*y - 1)
  MatType term1 = arma::pow(1.0 - p_t, gamma);
  MatType term2 = gamma * p_t % arma::log(p_t) + p_t - 1.0;
  MatType signFactor = 2.0 * target - 1.0;

  loss = alpha_t % term1 % term2 % signFactor;

  if (!reduction)
    loss /= prediction.n_elem;
}

template<typename MatType>
template<typename Archive>
void FocalLossType<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(gamma));
  ar(CEREAL_NVP(alpha));
  ar(CEREAL_NVP(reduction));
}

} // namespace mlpack

#endif
