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
  double penalty_threshold = DBL_MAX; // Ensure we update lambda immediately.

  // Track the last objective to compare for convergence.
  double last_objective = function.Evaluate(coordinates);

  // First, we create an instance of the utility function class.
  AugLagrangianFunction f(function, lambda, sigma);

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
    Log::Info << "AugLagrangian on iteration " << it
        << ", starting with objective "  << last_objective << "." << std::endl;

    // Use L-BFGS to optimize this function for the given lambda and sigma.
    L_BFGS<AugLagrangianFunction> lbfgs(f, numBasis);
    if (!lbfgs.Optimize(0, coordinates))
      Log::Info << "L-BFGS reported an error during optimization."
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
      penalty += std::pow(function.EvaluateConstraint(i, coordinates), 2);

    Log::Info << "Penalty is " << penalty << " (threshold "
        << penalty_threshold << ")." << std::endl;

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
      Log::Info << "Lagrange multiplier estimates updated." << std::endl;
    }
    else
    {
      // We multiply sigma by a constant value.  TODO: this factor should be a
      // parameter (from CLI).  The value of 10 is taken from Burer and Monteiro
      // (2002).
      sigma *= 10;
      f.Sigma() = sigma;
      Log::Info << "Updated sigma to " << sigma << "." << std::endl;
    }
  }

  return false;
}


template<typename LagrangianFunction>
AugLagrangian<LagrangianFunction>::AugLagrangianFunction::AugLagrangianFunction(
      LagrangianFunction& functionIn, arma::vec& lambdaIn, double sigma) :
    lambda(lambdaIn),
    sigma(sigma),
    function(functionIn)
{
  // Nothing to do.
}

template<typename LagrangianFunction>
double AugLagrangian<LagrangianFunction>::AugLagrangianFunction::Evaluate(
    const arma::mat& coordinates)
{
  // The augmented Lagrangian is evaluated as
  //   f(x) + {-lambdai * c_i(x) + (sigma / 2) c_i(x)^2} for all constraints
//  Log::Debug << "Evaluating augmented Lagrangian." << std::endl;
  double objective = function.Evaluate(coordinates);

  // Now loop over constraints.
  for (size_t i = 0; i < function.NumConstraints(); i++)
  {
    double constraint = function.EvaluateConstraint(i, coordinates);
    objective += (-lambda[i] * constraint) +
        sigma * std::pow(constraint, 2) / 2;
  }

//  Log::Warn << "Overall objective is " << objective << "." << std::endl;

  return objective;
}

template<typename LagrangianFunction>
void AugLagrangian<LagrangianFunction>::AugLagrangianFunction::Gradient(
    const arma::mat& coordinates, arma::mat& gradient)
{
  // The augmented Lagrangian's gradient is evaluated as
  // f'(x) + {(-lambdai + sigma * c_i(x)) * c'_i(x)} for all constraints
//  gradient.zeros();
  function.Gradient(coordinates, gradient);
//  Log::Debug << "Objective function gradient norm is "
//      << arma::norm(gradient, 2) << "." << std::endl;
//  std::cout << gradient << std::endl;

  arma::mat constraint_gradient; // Temporary for constraint gradients.
  for (size_t i = 0; i < function.NumConstraints(); i++)
  {
    function.GradientConstraint(i, coordinates, constraint_gradient);

    // Now calculate scaling factor and add to existing gradient.
    arma::mat tmp_gradient;
    tmp_gradient = (-lambda[i] + sigma *
        function.EvaluateConstraint(i, coordinates)) * constraint_gradient;
//    Log::Debug << "Gradient for constraint " << i << " (with lambda = "
//        << lambda[i] << ") is " << std::endl;
//    std::cout << tmp_gradient;
    gradient += tmp_gradient;
  }
//  Log::Debug << "Overall gradient norm is " << arma::norm(gradient, 2) << "."
//      << std::endl;
//  std::cout << gradient << std::endl;
}

template<typename LagrangianFunction>
const arma::mat& AugLagrangian<LagrangianFunction>::AugLagrangianFunction::
    GetInitialPoint() const
{
  return function.GetInitialPoint();
}

}; // namespace optimization
}; // namespace mlpack

#endif // __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP
