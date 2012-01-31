/**
 * @file lrsdp_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_LRSDP_LRSDP_IMPL_HPP

// In case it hasn't already been included.
#include "lrsdp.hpp"

// Augmented Lagrangian solver.
#include "../aug_lagrangian/aug_lagrangian.hpp"

namespace mlpack {
namespace optimization {

bool LRSDP::Optimize(arma::mat& coordinates)
{
  // Create the Augmented Lagrangian function.
  AugLagrangian<LRSDP> auglag(*this);

  auglag.Optimize(coordinates);
}

double LRSDP::Evaluate(const arma::mat& coordinates) const
{
  Log::Fatal << "LRSDP::Evaluate() called!  Uh-oh..." << std::endl;
}

void LRSDP::Gradient(const arma::mat& coordinates, arma::mat& gradient) const
{
  Log::Fatal << "LRSDP::Gradient() called!  Uh-oh..." << std::endl;
}

// Custom specializations of the AugmentedLagrangianFunction for the LRSDP case.
template<>
double AugLagrangianFunction<LRSDP>::Evaluate(const arma::mat& coordinates)
    const
{
  // We can calculate the entire objective in a smart way.
  // L(R, y, s) = Tr(C * (R R^T)) -
  //     sum_{i = 1}^{m} (y_i (Tr(A_i * (R R^T)) - b_i)) +
  //     (sigma / 2) * sum_{i = 1}^{m} (Tr(A_i * (R R^T)) - b_i)^2

  // Let's start with the objective: Tr(C * (R R^T)).
  // Simple, possibly slow solution.
  arma::mat rrt = coordinates * trans(coordinates);
  double objective = trace(function.C() * rrt);

  // Now each constraint.
  for (size_t i = 0; i < function.B().n_elem; ++i)
  {
    // Take the trace subtracted by the b_i.
    double constraint = trace(function.A()[i] * rrt) - function.B()[i];
    objective -= (lambda[i] * constraint);
    objective += (sigma / 2) * std::pow(constraint, 2.0);
  }

  return objective;
}

template<>
double AugLagrangianFunction<LRSDP>::Gradient(const arma::mat& coordinates,
                                              arma::mat& gradient) const
{
  // We can calculate the gradient in a smart way.
  // L'(R, y, s) = 2 * S' * R
  //   with
  // S' = C - sum_{i = 1}^{m} y'_i A_i
  // y'_i = y_i - sigma * (Trace(A_i * (R R^T)) - b_i)
  arma::mat s = function.C();

  for (size_t i = 0; i < function.B().n_elem; ++i)
  {
    double y = lambda[i] - sigma * (trace(function.A()[i] *
        (coordinates * trans(coordinates))) - function.B()[i]);
    s -= (y * function.A()[i]);
  }

  gradient = 2 * s * coordinates;
}

}; // namespace optimization
}; // namespace mlpack

#endif
