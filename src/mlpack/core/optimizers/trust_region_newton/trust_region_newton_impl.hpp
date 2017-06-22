/**
 * @file trust_region_newton_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Trust Region Newton Method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_TRUST_REGION_NEWTON_TRUST_REGION_NEWTON_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_TRUST_REGION_NEWTON_TRUST_REGION_NEWTON_IMPL_HPP

// In case it hasn't been included yet.
#include "trust_region_newton.hpp"

namespace mlpack {
namespace optimization {

template<typename FunctionType>
TrustRegionNewton<FunctionType>::TrustRegionNewton(
    FunctionType& function,
    const double minGradientNorm,
    const size_t maxIterations,
    const size_t maxConjugateIterations,
    const double eta0,
    const double eta1,
    const double eta2,
    const double sigma1,
    const double sigma2,
    const double sigma3) :
    function(function),
    minGradientNorm(minGradientNorm),
    maxIterations(maxIterations),
    maxConjugateIterations(maxConjugateIterations),
    eta0(eta0),
    eta1(eta1),
    eta2(eta2),
    sigma1(sigma1),
    sigma2(sigma2),
    sigma3(sigma3)
{ /* Nothing to do. */ }

template<typename FunctionType>
void TrustRegionNewton<FunctionType>::ConjugateGradient(
    FunctionType& function,
    const double delta,
    const arma::mat& iterate)
{
  // The initial gradient values.
  r = -gradient;
  arma::mat d = r;
  arma::mat hessian;

  s.zeros();
  double rTr = arma::dot(r, r);

  // Whether to optimize until convergence.
  const bool optimizeUntilConvergence = (maxConjugateIterations == 0);

  const double minGradientNormConjugate = 0.5 * arma::norm(gradient, 2);
  for (size_t itNum = 0; optimizeUntilConvergence ||
      (itNum != maxConjugateIterations); ++itNum)
  {
    // If ||r^i|| <= e_k||detla f(w^k)|| then output sk =s^(i) and stop.
    if (arma::norm(r, 2) <= minGradientNormConjugate)
      break;

    if (itNum > 0)
    {
      const double rTrUpdate = arma::dot(r, r);
      d *= (rTrUpdate / rTr);
      d += r;
      rTr = rTrUpdate;
    }

    function.Hessian(iterate, d, derivative, hessian);

    double alpha = rTr / arma::dot(d, hessian);
    s += alpha * d;
    if (arma::norm(s, 2) > delta)
    {
      // Compute tau such that ||s^(-i) + tau d^i|| = delta k.
      s -= alpha * d;

      const double std = arma::dot(s, d);
      const double sts = arma::dot(s, s);
      const double dtd = arma::dot(d, d);

      const double deltaSq = delta * delta;
      const double rad = std::sqrt(std * std + dtd * (deltaSq - sts));

      if (std >= 0)
        alpha = (deltaSq - sts) / (std + rad);
      else
        alpha = (rad - std) / dtd;

      // Output sk = s^(i) + tau d^i and stop.
      s += alpha * d;
      r -= alpha * hessian;
      break;
    }

    r -= alpha * hessian;
    // break;
  }
}

template<typename FunctionType>
double TrustRegionNewton<FunctionType>::Optimize(
    FunctionType& function, arma::mat& iterate)
{
  // The initial function value.
  double functionValue = function.Evaluate(iterate);

  // The initial gradient value.
  function.Gradient(iterate, gradient, derivative);
  double delta = arma::norm(gradient, 2);

  // To keep track of where we are and how things are going.
  size_t k = 0;
  double alpha = 0;
  s.set_size(gradient.n_rows, gradient.n_cols);

  // Whether to optimize until convergence.
  const bool optimizeUntilConvergence = (maxIterations == 0);

  // The main optimization loop.
  for (size_t itNum = 0; optimizeUntilConvergence || (itNum != maxIterations);
       ++itNum)
  {
    // Break when the norm of the gradient becomes too small.
    //
    // But don't do this on the first iteration to ensure we always take at
    // least one descent step.
    if (itNum > 0 && arma::norm(gradient, 2) < minGradientNorm)
    {
      Log::Debug << "Trust Region Newton gradient norm too small "
          << "(terminating successfully)." << std::endl;
      std::cout << "okay\n";
      break;
    }

    // Break if the objective is not a number.
    if (std::isnan(functionValue))
    {
      Log::Warn << "Trust Region Newton terminated with objective "
          << functionValue << "; " << "are the objective and gradient "
          << "functions implemented correctly?" << std::endl;
      break;
    }

    // Find an approximate solution s^k of the trust region sub-problem.
    ConjugateGradient(function, delta, iterate);

    // Save the old iterate before stepping.
    arma::mat newIterate = s + iterate;

    // It is possible that the difference between the two coordinates is zero.
    // In this case we terminate successfully.
    if (arma::accu(iterate != newIterate) == 0)
    {
      Log::Debug << "Trust Region Newton step size of 0 "
          << "(terminating successfully)." << std::endl;
      break;
    }

    double gs = arma::dot(gradient, s);
    double denom = -0.5 * (gs - arma::dot(s, r));

    const double newFunctionValue = function.Evaluate(newIterate);
    const double nume = functionValue - newFunctionValue;
    const double snorm = arma::norm(s, 2);

    if (k == 1)
      delta = std::min(delta, snorm);

    if (newFunctionValue - functionValue - gs <= 0)
    {
      alpha = sigma3;
    }
    else
    {
      alpha = std::max(sigma1, -0.5 *
          (gs / (newFunctionValue - functionValue - gs)));
    }

    if (nume < eta0 * denom)
      delta = std::min(std::max(alpha, sigma1) * snorm, sigma2 * delta);
    else if (nume < eta1 * denom)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma2 * delta));
    else if (nume < eta2 * denom)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma3 * delta));
    else
      delta = std::max(delta, std::min(alpha * snorm, sigma3 * delta));

    if (nume > eta0 * denom)
    {
      k += 1;
      // Update the old iterate.
      iterate = newIterate;
      functionValue = newFunctionValue;
      function.Gradient(iterate, gradient, derivative);
    }
  } // End of the optimization loop.

  return function.Evaluate(iterate);
}

} // namespace optimization
} // namespace mlpack

#endif
