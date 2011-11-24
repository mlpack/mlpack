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
      LagrangianFunction& function_in, int num_basis) :
    function_(function_in),
    num_basis_(num_basis)
{
  // Not sure what to do here (if anything).
}

template<typename LagrangianFunction>
bool AugLagrangian<LagrangianFunction>::Optimize(int num_iterations,
                                                 arma::mat& coordinates,
                                                 double sigma)
{
  // Choose initial lambda parameters (vector of zeros, for simplicity).
  arma::vec lambda(function_.NumConstraints());
  lambda.ones();
  double penalty_threshold = DBL_MAX; // Ensure we update lambda immediately.

  // Track the last objective to compare for convergence.
  double last_objective = function_.Evaluate(coordinates);

  // First, we create an instance of the utility function class.
  AugLagrangianFunction f(function_, lambda, sigma);

  // First, calculate the current penalty.
  double penalty = 0;
  for (int i = 0; i < function_.NumConstraints(); i++)
    penalty += std::pow(function_.EvaluateConstraint(i, coordinates), 2);

  Log::Debug << "Penalty is " << penalty << " (threshold " << penalty_threshold
      << ")." << std::endl;

  // The odd comparison allows user to pass num_iterations = 0 (i.e. no limit on
  // number of iterations).
  int it;
  for (it = 0; it != (num_iterations - 1); it++)
  {
    Log::Debug << "AugLagrangian on iteration " << it
        << ", starting with objective "  << last_objective << "." << std::endl;

    // Use L-BFGS to optimize this function for the given lambda and sigma.
    L_BFGS<AugLagrangianFunction> lbfgs(f, num_basis_);
    if (!lbfgs.Optimize(0, coordinates))
      Log::Debug << "L-BFGS reported an error during optimization."
          << std::endl;

    // Check if we are done with the entire optimization (the threshold we are
    // comparing with is arbitrary).
    if (std::abs(last_objective - function_.Evaluate(coordinates)) < 1e-10 &&
        sigma > 500000)
      return true;

    last_objective = function_.Evaluate(coordinates);

    // Assuming that the optimization has converged to a new set of coordinates,
    // we now update either lambda or sigma.  We update sigma if the penalty
    // term is too high, and we update lambda otherwise.

    // First, calculate the current penalty.
    double penalty = 0;
    for (int i = 0; i < function_.NumConstraints(); i++)
      penalty += std::pow(function_.EvaluateConstraint(i, coordinates), 2);

    Log::Debug << "Penalty is " << penalty << " (threshold "
        << penalty_threshold << ")." << std::endl;
    if (penalty < penalty_threshold) // We update lambda.
    {
      // We use the update: lambda_{k + 1} = lambda_k - sigma * c(coordinates),
      // but we have to write a loop to do this for each constraint.
      for (int i = 0; i < function_.NumConstraints(); i++)
        lambda[i] -= sigma * function_.EvaluateConstraint(i, coordinates);
      f.lambda_ = lambda;

      // We also update the penalty threshold to be a factor of the current
      // penalty.  TODO: this factor should be a parameter (from CLI).  The
      // value of 0.25 is taken from Burer and Monteiro (2002).
      penalty_threshold = 0.25 * penalty;
      Log::Debug << "Lagrange multiplier estimates updated." << std::endl;
    }
    else
    {
      // We multiply sigma by a constant value.  TODO: this factor should be a
      // parameter (from CLI).  The value of 10 is taken from Burer and Monteiro
      // (2002).
      sigma *= 10;
      f.sigma_ = sigma;
      Log::Debug << "Updated sigma to " << sigma << "." << std::endl;
    }
  }

  return false;
}


template<typename LagrangianFunction>
AugLagrangian<LagrangianFunction>::AugLagrangianFunction::AugLagrangianFunction(
      LagrangianFunction& function_in, arma::vec& lambda_in, double sigma) :
    lambda_(lambda_in),
    sigma_(sigma),
    function_(function_in)
{
  // Nothing to do.
}

template<typename LagrangianFunction>
double AugLagrangian<LagrangianFunction>::AugLagrangianFunction::Evaluate(
    const arma::mat& coordinates)
{
  // The augmented Lagrangian is evaluated as
  //   f(x) + {-lambda_i * c_i(x) + (sigma / 2) c_i(x)^2} for all constraints
//  Log::Debug << "Evaluating augmented Lagrangian." << std::endl;
  double objective = function_.Evaluate(coordinates);

  // Now loop over constraints.
  for (int i = 0; i < function_.NumConstraints(); i++)
  {
    double constraint = function_.EvaluateConstraint(i, coordinates);
    objective += (-lambda_[i] * constraint) +
        sigma_ * std::pow(constraint, 2) / 2;
  }

//  Log::Warn << "Overall objective is " << objective << "." << std::endl;

  return objective;
}

template<typename LagrangianFunction>
void AugLagrangian<LagrangianFunction>::AugLagrangianFunction::Gradient(
    const arma::mat& coordinates, arma::mat& gradient)
{
  // The augmented Lagrangian's gradient is evaluated as
  // f'(x) + {(-lambda_i + sigma * c_i(x)) * c'_i(x)} for all constraints
//  gradient.zeros();
  function_.Gradient(coordinates, gradient);
//  Log::Debug << "Objective function gradient norm is "
//      << arma::norm(gradient, 2) << "." << std::endl;
//  std::cout << gradient << std::endl;

  arma::mat constraint_gradient; // Temporary for constraint gradients.
  for (int i = 0; i < function_.NumConstraints(); i++)
  {
    function_.GradientConstraint(i, coordinates, constraint_gradient);

    // Now calculate scaling factor and add to existing gradient.
    arma::mat tmp_gradient;
    tmp_gradient = (-lambda_[i] + sigma_ *
        function_.EvaluateConstraint(i, coordinates)) * constraint_gradient;
//    Log::Debug << "Gradient for constraint " << i << " (with lambda = "
//        << lambda_[i] << ") is " << std::endl;
//    std::cout << tmp_gradient;
    gradient += tmp_gradient;
  }
//  Log::Debug << "Overall gradient norm is " << arma::norm(gradient, 2) << "."
//      << std::endl;
//  std::cout << gradient << std::endl;
}

template<typename LagrangianFunction>
const arma::mat& AugLagrangian<LagrangianFunction>::AugLagrangianFunction::
    GetInitialPoint()
{
  return function_.GetInitialPoint();
}

}; // namespace optimization
}; // namespace mlpack

#endif // __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP
