/**
 * @file lrsdp_function.cpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * Implementation of the LRSDPFunction class, and also template specializations
 * for faster execution with the AugLagrangian optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_FUNCTION_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_LRSDP_FUNCTION_IMPL_HPP

#include "lrsdp_function.hpp"

namespace mlpack {
namespace optimization {

template <typename SDPType>
LRSDPFunction<SDPType>::LRSDPFunction(const SDPType& sdp,
                                      const arma::mat& initialPoint):
    sdp(sdp),
    initialPoint(initialPoint)
{
  if (initialPoint.n_rows < initialPoint.n_cols)
    Log::Warn << "LRSDPFunction::LRSDPFunction(): solution matrix will have "
        << "more columns than rows.  It may be more efficient to find the "
        << "transposed solution." << std::endl;
}

template <typename SDPType>
LRSDPFunction<SDPType>::LRSDPFunction(const size_t numSparseConstraints,
                                      const size_t numDenseConstraints,
                                      const arma::mat& initialPoint):
    sdp(initialPoint.n_rows, numSparseConstraints, numDenseConstraints),
    initialPoint(initialPoint)
{
  if (initialPoint.n_rows < initialPoint.n_cols)
    Log::Warn << "LRSDPFunction::LRSDPFunction(): solution matrix will have "
        << "more columns than rows.  It may be more efficient to find the "
        << "transposed solution." << std::endl;
}

template <typename SDPType>
double LRSDPFunction<SDPType>::Evaluate(const arma::mat& coordinates) const
{
  const arma::mat rrt = coordinates * trans(coordinates);
  return accu(SDP().C() % rrt);
}

template <typename SDPType>
void LRSDPFunction<SDPType>::Gradient(const arma::mat& /* coordinates */,
                                      arma::mat& /* gradient */) const
{
  Log::Fatal << "LRSDPFunction::Gradient() not implemented for arbitrary optimizers!"
      << std::endl;
}

template <typename SDPType>
double LRSDPFunction<SDPType>::EvaluateConstraint(const size_t index,
                                                  const arma::mat& coordinates) const
{
  const arma::mat rrt = coordinates * trans(coordinates);
  if (index < SDP().NumSparseConstraints())
    return accu(SDP().SparseA()[index] % rrt) - SDP().SparseB()[index];
  const size_t index1 = index - SDP().NumSparseConstraints();
  return accu(SDP().DenseA()[index1] % rrt) - SDP().DenseB()[index1];
}

template <typename SDPType>
void LRSDPFunction<SDPType>::GradientConstraint(const size_t /* index */,
                                                const arma::mat& /* coordinates */,
                                                arma::mat& /* gradient */) const
{
  Log::Fatal << "LRSDPFunction::GradientConstraint() not implemented for arbitrary "
      << "optimizers!" << std::endl;
}

//! Utility function for calculating part of the objective when AugLagrangian is
//! used with an LRSDPFunction.
template <typename MatrixType>
static inline void
UpdateObjective(double& objective,
                const arma::mat& rrt,
                const std::vector<MatrixType>& ais,
                const arma::vec& bis,
                const arma::vec& lambda,
                const size_t lambdaOffset,
                const double sigma)
{
  for (size_t i = 0; i < ais.size(); ++i)
  {
    // Take the trace subtracted by the b_i.
    const double constraint = accu(ais[i] % rrt) - bis[i];
    objective -= (lambda[lambdaOffset + i] * constraint);
    objective += (sigma / 2.) * constraint * constraint;
  }
}

//! Utility function for calculating part of the gradient when AugLagrangian is
//! used with an LRSDPFunction.
template <typename MatrixType>
static inline void
UpdateGradient(arma::mat& s,
               const arma::mat& rrt,
               const std::vector<MatrixType>& ais,
               const arma::vec& bis,
               const arma::vec& lambda,
               const size_t lambdaOffset,
               const double sigma)
{
  for (size_t i = 0; i < ais.size(); ++i)
  {
    const double constraint = accu(ais[i] % rrt) - bis[i];
    const double y = lambda[lambdaOffset + i] - sigma * constraint;
    s -= y * ais[i];
  }
}

template <typename SDPType>
static inline double
EvaluateImpl(const LRSDPFunction<SDPType>& function,
             const arma::mat& coordinates,
             const arma::vec& lambda,
             const double sigma)
{
  // We can calculate the entire objective in a smart way.
  // L(R, y, s) = Tr(C * (R R^T)) -
  //     sum_{i = 1}^{m} (y_i (Tr(A_i * (R R^T)) - b_i)) +
  //     (sigma / 2) * sum_{i = 1}^{m} (Tr(A_i * (R R^T)) - b_i)^2

  // Let's start with the objective: Tr(C * (R R^T)).
  // Simple, possibly slow solution-- see below for optimization opportunity
  //
  // TODO: Note that Tr(C^T * (R R^T)) = Tr( (CR)^T * R ), so
  // multiplying C*R first, and then taking the trace dot should be more memory
  // efficient
  //
  // Similarly for the constraints, taking A*R first should be more efficient
  const arma::mat rrt = coordinates * trans(coordinates);
  double objective = accu(function.SDP().C() % rrt);

  // Now each constraint.
  UpdateObjective(objective, rrt, function.SDP().SparseA(), function.SDP().SparseB(),
      lambda, 0, sigma);
  UpdateObjective(objective, rrt, function.SDP().DenseA(), function.SDP().DenseB(), lambda,
      function.SDP().NumSparseConstraints(), sigma);

  return objective;
}

template <typename SDPType>
static inline void
GradientImpl(const LRSDPFunction<SDPType>& function,
             const arma::mat& coordinates,
             const arma::vec& lambda,
             const double sigma,
             arma::mat& gradient)
{
  // We can calculate the gradient in a smart way.
  // L'(R, y, s) = 2 * S' * R
  //   with
  // S' = C - sum_{i = 1}^{m} y'_i A_i
  // y'_i = y_i - sigma * (Trace(A_i * (R R^T)) - b_i)
  const arma::mat rrt = coordinates * trans(coordinates);
  arma::mat s(function.SDP().C());

  UpdateGradient(
      s, rrt, function.SDP().SparseA(), function.SDP().SparseB(),
      lambda, 0, sigma);
  UpdateGradient(
      s, rrt, function.SDP().DenseA(), function.SDP().DenseB(),
      lambda, function.SDP().NumSparseConstraints(), sigma);

  gradient = 2 * s * coordinates;
}

// Template specializations for function and gradient evaluation.
// Note that C++ does not allow partial specialization of class members,
// so we have to go about this in a somewhat round-about way.
template <>
inline double AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Evaluate(
    const arma::mat& coordinates) const
{
  return EvaluateImpl(function, coordinates, lambda, sigma);
}

template <>
inline double AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Evaluate(
    const arma::mat& coordinates) const
{
  return EvaluateImpl(function, coordinates, lambda, sigma);
}

template <>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::sp_mat>>>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const
{
  GradientImpl(function, coordinates, lambda, sigma, gradient);
}

template <>
inline void AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const
{
  GradientImpl(function, coordinates, lambda, sigma, gradient);
}

} // namespace optimization
} // namespace mlpack

#endif
