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

namespace mlpack {
namespace optimization {

// convert the object to a string
std::string LRSDP::ToString() const{
  std::ostringstream convert;
  convert << "LRSDP [" << this << "]" << std::endl;
  convert << "Matrix Size: " << c.n_rows << "x" << c.n_cols << std::endl;
  convert << "Initial point Size : " << initialPoint.n_rows << "x" 
      << initialPoint.n_cols << std::endl;
  //convert << "AugLagrangian Info: " << std::endl << augLag;
  return convert.str();
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
    double constraint = -function.B()[i];

    if (function.AModes()[i] == 0)
    {
      constraint += trace(function.A()[i] * rrt);
    }
    else
    {
      for (size_t j = 0; j < function.A()[i].n_cols; ++j)
      {
        constraint += function.A()[i](2, j) *
            rrt(function.A()[i](0, j), function.A()[i](1, j));
      }
    }

    objective -= (lambda[i] * constraint);
    objective += (sigma / 2) * std::pow(constraint, 2.0);
  }

  return objective;
}

template<>
void AugLagrangianFunction<LRSDP>::Gradient(const arma::mat& coordinates,
                                            arma::mat& gradient) const
{
  // We can calculate the gradient in a smart way.
  // L'(R, y, s) = 2 * S' * R
  //   with
  // S' = C - sum_{i = 1}^{m} y'_i A_i
  // y'_i = y_i - sigma * (Trace(A_i * (R R^T)) - b_i)
  arma::mat rrt = coordinates * trans(coordinates);
  arma::mat s = function.C();

  for (size_t i = 0; i < function.B().n_elem; ++i)
  {
    double constraint = -function.B()[i];

    if (function.AModes()[i] == 0)
    {
      constraint += trace(function.A()[i] * rrt);
    }
    else
    {
      for (size_t j = 0; j < function.A()[i].n_cols; ++j)
      {
        constraint += function.A()[i](2, j) *
            rrt(function.A()[i](0, j), function.A()[i](1, j));
      }
    }

    double y = lambda[i] - sigma * constraint;

    if (function.AModes()[i] == 0)
    {
      s -= (y * function.A()[i]);
    }
    else
    {
      // We only need to subtract the entries which could be modified.
      for (size_t j = 0; j < function.A()[i].n_cols; ++j)
      {
        s(function.A()[i](0, j), function.A()[i](1, j)) -= y;
      }
    }
  }

  gradient = 2 * s * coordinates;
}

}; // namespace optimization
}; // namespace mlpack

#endif

