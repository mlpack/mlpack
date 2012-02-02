/**
 * @file aug_lagrangian_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangian class (Augmented Lagrangian optimization
 * method).
 */

#ifndef __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP

#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include "aug_lagrangian_function.hpp"

namespace mlpack {
namespace optimization {

template<typename LagrangianFunction>
AugLagrangian<LagrangianFunction>::AugLagrangian(
      LagrangianFunction& function, size_t numBasis) :
    function(function),
    numBasis(numBasis)
{
  // Not sure what to do here (if anything).
}

template<typename LagrangianFunction>
bool AugLagrangian<LagrangianFunction>::Optimize(arma::mat& coordinates,
                                                 const size_t maxIterations,
                                                 double sigma)
{
  // Choose initial lambda parameters (vector of zeros, for simplicity).
  arma::vec lambda(function.NumConstraints());
  lambda.ones();
  lambda *= -1;
  lambda[0] = -0.70 * double(coordinates.n_cols);
  double penalty_threshold = DBL_MAX; // Ensure we update lambda immediately.

  // Track the last objective to compare for convergence.
  double last_objective = function.Evaluate(coordinates);

  // First, we create an instance of the utility function class.
  AugLagrangianFunction<LagrangianFunction> f(function, lambda, sigma);

  // First, calculate the current penalty.
  double penalty = 0;
  for (size_t i = 0; i < function.NumConstraints(); i++)
    penalty += std::pow(function.EvaluateConstraint(i, coordinates), 2);

  Log::Info << "Penalty is " << penalty << " (threshold " << penalty_threshold
      << ")." << std::endl;

  // The odd comparison allows user to pass maxIterations = 0 (i.e. no limit on
  // number of iterations).
  size_t it;
  for (it = 0; it != (maxIterations - 1); it++)
  {
    Log::Warn << "AugLagrangian on iteration " << it
        << ", starting with objective "  << last_objective << "." << std::endl;

 //   Log::Warn << coordinates << std::endl;

//    Log::Warn << trans(coordinates) * coordinates << std::endl;

    // Use L-BFGS to optimize this function for the given lambda and sigma.
    L_BFGS<AugLagrangianFunction<LagrangianFunction> >
        lbfgs(f, numBasis, 1e-4, 0.9, 1e-10, 100, 1e-20, 1e20);

    if (!lbfgs.Optimize(0, coordinates))
      Log::Warn << "L-BFGS reported an error during optimization."
          << std::endl;

    // Check if we are done with the entire optimization (the threshold we are
    // comparing with is arbitrary).
    if (std::abs(last_objective - function.Evaluate(coordinates)) < 1e-10 &&
        sigma > 500000)
      return true;

    last_objective = function.Evaluate(coordinates);

    // Assuming that the optimization has converged to a new set of coordinates,
    // we now update either lambda or sigma.  We update sigma if the penalty
    // term is too high, and we update lambda otherwise.

    // First, calculate the current penalty.
    double penalty = 0;
    for (size_t i = 0; i < function.NumConstraints(); i++)
    {
      penalty += std::pow(function.EvaluateConstraint(i, coordinates), 2);
//      Log::Debug << "Constraint " << i << " is " <<
//          function.EvaluateConstraint(i, coordinates) << std::endl;
    }

    Log::Warn << "Penalty is " << penalty << " (threshold "
        << penalty_threshold << ")." << std::endl;

    for (size_t i = 0; i < function.NumConstraints(); ++i)
    {
//      arma::mat tmpgrad;
//      function.GradientConstraint(i, coordinates, tmpgrad);
//      Log::Debug << "Gradient of constraint " << i << " is " << std::endl;
//      Log::Debug << tmpgrad << std::endl;
    }

    if (penalty < penalty_threshold) // We update lambda.
    {
      // We use the update: lambda{k + 1} = lambdak - sigma * c(coordinates),
      // but we have to write a loop to do this for each constraint.
      for (size_t i = 0; i < function.NumConstraints(); i++)
        lambda[i] -= sigma * function.EvaluateConstraint(i, coordinates);
      f.Lambda() = lambda;

      // We also update the penalty threshold to be a factor of the current
      // penalty.  TODO: this factor should be a parameter (from CLI).  The
      // value of 0.25 is taken from Burer and Monteiro (2002).
      penalty_threshold = 0.25 * penalty;
      Log::Warn << "Lagrange multiplier estimates updated." << std::endl;
    }
    else
    {
      // We multiply sigma by a constant value.  TODO: this factor should be a
      // parameter (from CLI).  The value of 10 is taken from Burer and Monteiro
      // (2002).
      sigma *= 10;
      f.Sigma() = sigma;
      Log::Warn << "Updated sigma to " << sigma << "." << std::endl;
    }
  }

  return false;
}

}; // namespace optimization
}; // namespace mlpack

#endif // __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP
