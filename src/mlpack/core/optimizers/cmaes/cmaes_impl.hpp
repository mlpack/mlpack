/**
 * @file cmaes_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Implementation of the Covariance Matrix Adaptation Evolution Strategy as
 * proposed by N. Hansen et al. in "Completely Derandomized Self-Adaptation in
 * Evolution Strategies".
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP

// In case it hasn't been included yet.
#include "cmaes.hpp"

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template<typename SelectionPolicyType>
CMAES<SelectionPolicyType>::CMAES(const size_t lambda,
                                  const double lowerBound,
                                  const double upperBound,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const SelectionPolicyType& selectionPolicy) :
    lambda(lambda),
    lowerBound(lowerBound),
    upperBound(upperBound),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    selectionPolicy(selectionPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename SelectionPolicyType>
template<typename DecomposableFunctionType>
double CMAES<SelectionPolicyType>::Optimize(
    DecomposableFunctionType& function, arma::mat& iterate)
{
  // Make sure that we have the methods that we need.  Long name...
  traits::CheckNonDifferentiableDecomposableFunctionTypeAPI<
      DecomposableFunctionType>();

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // Population size.
  if (lambda == 0)
    lambda = (4 + std::round(3 * std::log(iterate.n_elem))) * 10;

  // Parent weights.
  const size_t mu = std::round(lambda / 2);
  arma::vec w = std::log(mu + 0.5) - arma::log(
    arma::linspace<arma::vec>(0, mu - 1, mu) + 1.0);
  w /= arma::sum(w);

  // Number of effective solutions.
  const double muEffective = 1 / arma::accu(arma::pow(w, 2));

  // Step size control parameters.
  arma::vec sigma(3);
  sigma(0) = 0.3 * (upperBound - lowerBound);
  const double cs = (muEffective + 2) / (iterate.n_elem + muEffective + 5);
  const double ds = 1 + cs + 2 * std::max(std::sqrt((muEffective - 1) /
      (iterate.n_elem + 1)) - 1, 0.0);
  const double enn = std::sqrt(iterate.n_elem) * (1.0 - 1.0 /
      (4.0 * iterate.n_elem) + 1.0 / (21 * std::pow(iterate.n_elem, 2)));

  // Covariance update parameters.
  // Cumulation for distribution.
  const double cc = (4 + muEffective / iterate.n_elem) /
      (4 + iterate.n_elem + 2 * muEffective / iterate.n_elem);
  const double h = (1.4 + 2.0 / (iterate.n_elem + 1.0)) * enn;

  const double c1 = 2 / (std::pow(iterate.n_elem + 1.3, 2) + muEffective);
  const double alphaMu = 2;
  const double cmu = std::min(1 - c1, alphaMu * (muEffective - 2 + 1 /
      muEffective) / (std::pow(iterate.n_elem + 2, 2) +
      alphaMu * muEffective / 2));

  arma::cube mPosition(iterate.n_rows, iterate.n_cols, 3);
  mPosition.slice(0) = lowerBound + arma::randu(
      iterate.n_rows, iterate.n_cols) * (upperBound - lowerBound);

  arma::mat step = arma::zeros(iterate.n_rows, iterate.n_cols);

  // Calculate the first objective function.
  double currentObjective = 0;
  for (size_t f = 0; f < numFunctions; f += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
    currentObjective += function.Evaluate(mPosition.slice(0), f,
        effectiveBatchSize);
  }

  double overallObjective = currentObjective;
  double lastObjective = DBL_MAX;

  // Population parameters.
  arma::cube pStep(iterate.n_rows, iterate.n_cols, lambda);
  arma::cube pPosition(iterate.n_rows, iterate.n_cols, lambda);
  arma::vec pObjective(lambda);
  arma::cube ps = arma::zeros(iterate.n_rows, iterate.n_cols, 2);
  arma::cube pc = ps;
  arma::cube C(iterate.n_elem, iterate.n_elem, 2);
  C.slice(0).eye();

  // Covariance matrix parameters.
  arma::vec eigval;
  arma::mat eigvec;
  arma::vec eigvalZero = arma::zeros(iterate.n_elem);

  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, lambda - 1, lambda);

  // Now iterate!
  for (size_t i = 1; i < maxIterations; ++i)
  {
    // To keep track of where we are.
    const size_t idx0 = (i - 1) % 2;
    const size_t idx1 = i % 2;

    const arma::mat covLower = arma::chol(C.slice(idx0), "lower");

    for (size_t j = 0; j < lambda; ++j)
    {
      if (iterate.n_rows > iterate.n_cols)
      {
        pStep.slice(idx(j)) = covLower *
            arma::randn(iterate.n_rows, iterate.n_cols);
      }
      else
      {
        pStep.slice(idx(j)) = arma::randn(iterate.n_rows, iterate.n_cols) *
            covLower;
      }

      pPosition.slice(idx(j)) = mPosition.slice(idx0) + sigma(idx0) *
          pStep.slice(idx(j));

      // Calculate the objective function.
      pObjective(idx(j)) = selectionPolicy.Select(function, batchSize,
          pPosition.slice(idx(j)));
    }

    // Sort population.
    idx = sort_index(pObjective);

    step = w(0) * pStep.slice(idx(0));
    for (size_t j = 1; j < mu; ++j)
      step += w(j) * pStep.slice(idx(j));

    mPosition.slice(idx1) = mPosition.slice(idx0) + sigma(idx0) * step;

    // Calculate the objective function.
    currentObjective = selectionPolicy.Select(function, batchSize,
          mPosition.slice(idx1));

    // Update best parameters.
    if (currentObjective < overallObjective)
    {
      overallObjective = currentObjective;
      iterate = mPosition.slice(idx1);
    }

    // Update Step Size.
    if (iterate.n_rows > iterate.n_cols)
    {
      ps.slice(idx1) = (1 - cs) * ps.slice(idx0) + std::sqrt(
          cs * (2 - cs) * muEffective) * covLower.t() * step;
    }
    else
    {
      ps.slice(idx1) = (1 - cs) * ps.slice(idx0) + std::sqrt(
          cs * (2 - cs) * muEffective) * step * covLower.t();
    }

    const double psNorm = arma::norm(ps.slice(idx1));
    sigma(idx1) = sigma(idx0) * std::pow(
        std::exp(cs / ds * psNorm / enn - 1), 0.3);

    // Update covariance matrix.
    if ((psNorm / sqrt(1 - std::pow(1 - cs, 2 * i))) < h)
    {
      pc.slice(idx1) = (1 - cc) * pc.slice(idx0) + std::sqrt(cc * (2 - cc) *
        muEffective) * step;


      if (iterate.n_rows > iterate.n_cols)
      {
        C.slice(idx1) = (1 - c1 - cmu) * C.slice(idx0) + c1 *
          (pc.slice(idx1) * pc.slice(idx1).t());
      }
      else
      {
        C.slice(idx1) = (1 - c1 - cmu) * C.slice(idx0) + c1 *
          (pc.slice(idx1).t() * pc.slice(idx1));
      }
    }
    else
    {
      pc.slice(idx1) = (1 - cc) * pc.slice(idx0);

      if (iterate.n_rows > iterate.n_cols)
      {
        C.slice(idx1) = (1 - c1 - cmu) * C.slice(idx0) + c1 * (pc.slice(idx1) *
            pc.slice(idx1).t() + (cc * (2 - cc)) * C.slice(idx0));
      }
      else
      {
        C.slice(idx1) = (1 - c1 - cmu) * C.slice(idx0) + c1 *
            (pc.slice(idx1).t() * pc.slice(idx1) + (cc * (2 - cc)) *
            C.slice(idx0));
      }
    }

    if (iterate.n_rows > iterate.n_cols)
    {
      for (size_t j = 0; j < mu; ++j)
      {
        C.slice(idx1) = C.slice(idx1) + cmu * w(j) *
            pStep.slice(idx(j)) * pStep.slice(idx(j)).t();
      }
    }
    else
    {
      for (size_t j = 0; j < mu; ++j)
      {
        C.slice(idx1) = C.slice(idx1) + cmu * w(j) *
            pStep.slice(idx(j)).t() * pStep.slice(idx(j));
      }
    }

    arma::eig_sym(eigval, eigvec, C.slice(idx1));
    const arma::uvec negativeEigval = find(eigval < 0, 1);
    if (!negativeEigval.is_empty())
    {
      if (negativeEigval(0) == 0)
      {
        C.slice(idx1).zeros();
      }
      else
      {
        C.slice(idx1) = eigvec.cols(0, negativeEigval(0) - 1) *
            arma::diagmat(eigval.subvec(0, negativeEigval(0) - 1)) *
            eigvec.cols(0, negativeEigval(0) - 1).t();
      }
    }

    // Output current objective function.
    Log::Info << "CMA-ES: iteration " << i << ", objective " << overallObjective
        << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Log::Warn << "CMA-ES: converged to " << overallObjective << "; "
          << "terminating with failure.  Try a smaller step size?" << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Log::Info << "CMA-ES: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;
      return overallObjective;
    }

    lastObjective = overallObjective;
  }

  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
