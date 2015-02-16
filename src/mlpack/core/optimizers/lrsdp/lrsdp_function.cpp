/**
 * @file lrsdp_function.cpp
 * @author Ryan Curtin
 * @author Abhishek Laddha
 *
 * Implementation of the LRSDPFunction class, and also template specializations
 * for faster execution with the AugLagrangian optimizer.
 */
#include "lrsdp_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace std;

LRSDPFunction::LRSDPFunction(const size_t numSparseConstraints,
                             const size_t numDenseConstraints,
                             const arma::mat& initialPoint):
    sparseC(initialPoint.n_rows, initialPoint.n_rows),
    denseC(initialPoint.n_rows, initialPoint.n_rows),
    hasModifiedSparseObjective(false),
    hasModifiedDenseObjective(false),
    sparseA(numSparseConstraints),
    sparseB(numSparseConstraints),
    denseA(numDenseConstraints),
    denseB(numDenseConstraints),
    initialPoint(initialPoint)
{
  denseC.zeros();
  if (initialPoint.n_rows < initialPoint.n_cols)
    Log::Warn << "LRSDPFunction::LRSDPFunction(): solution matrix will have "
        << "more columns than rows.  It may be more efficient to find the "
        << "transposed solution." << endl;
}

double LRSDPFunction::Evaluate(const arma::mat& coordinates) const
{
  const arma::mat rrt = coordinates * trans(coordinates);
  double objective = 0.;
  if (hasSparseObjective())
    objective += trace(SparseC() * rrt);
  if (hasDenseObjective())
    objective += trace(DenseC() * rrt);
  return objective;
}

void LRSDPFunction::Gradient(const arma::mat& /* coordinates */,
                     arma::mat& /* gradient */) const
{
  Log::Fatal << "LRSDP::Gradient() not implemented for arbitrary optimizers!"
      << std::endl;
}

double LRSDPFunction::EvaluateConstraint(const size_t index,
                                 const arma::mat& coordinates) const
{
  const arma::mat rrt = coordinates * trans(coordinates);
  if (index < NumSparseConstraints())
    return trace(sparseA[index] * rrt) - sparseB[index];
  const size_t index1 = index - NumSparseConstraints();
  return trace(denseA[index1] * rrt) - denseB[index1];
}

void LRSDPFunction::GradientConstraint(const size_t /* index */,
                               const arma::mat& /* coordinates */,
                               arma::mat& /* gradient */) const
{
  Log::Fatal << "LRSDP::GradientConstraint() not implemented for arbitrary "
      << "optimizers!" << std::endl;
}

// Return a string representation of the object.
std::string LRSDPFunction::ToString() const
{
  std::ostringstream convert;
  convert << "LRSDPFunction [" << this << "]" << std::endl;
  convert << "  Number of constraints: " << NumConstraints() << std::endl;
  convert << "  Problem size: n=" << initialPoint.n_rows << ", r="
      << initialPoint.n_cols << std::endl;
  convert << "  Sparse Constraint b_i values: " << sparseB.t();
  convert << "  Dense Constraint b_i values: " << denseB.t();
  return convert.str();
}

namespace mlpack {
namespace optimization {

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
    const double constraint = trace(ais[i] * rrt) - bis[i];
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
    const double constraint = trace(ais[i] * rrt) - bis[i];
    const double y = lambda[lambdaOffset + i] - sigma * constraint;
    s -= y * ais[i];
  }
}

// Template specializations for function and gradient evaluation.
template<>
double AugLagrangianFunction<LRSDPFunction>::Evaluate(
    const arma::mat& coordinates) const
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
  double objective = 0.;
  if (function.hasSparseObjective())
    objective += trace(function.SparseC() * rrt);
  if (function.hasDenseObjective())
    objective += trace(function.DenseC() * rrt);

  // Now each constraint.
  UpdateObjective(objective, rrt, function.SparseA(), function.SparseB(),
      lambda, 0, sigma);
  UpdateObjective(objective, rrt, function.DenseA(), function.DenseB(), lambda,
      function.NumSparseConstraints(), sigma);

  return objective;
}


template<>
void AugLagrangianFunction<LRSDPFunction>::Gradient(
    const arma::mat& coordinates,
    arma::mat& gradient) const
{
  // We can calculate the gradient in a smart way.
  // L'(R, y, s) = 2 * S' * R
  //   with
  // S' = C - sum_{i = 1}^{m} y'_i A_i
  // y'_i = y_i - sigma * (Trace(A_i * (R R^T)) - b_i)
  const arma::mat rrt = coordinates * trans(coordinates);
  arma::mat s(function.n(), function.n());
  s.zeros();

  if (function.hasSparseObjective())
    s += function.SparseC();
  if (function.hasDenseObjective())
    s += function.DenseC();

  UpdateGradient(
      s, rrt, function.SparseA(), function.SparseB(),
      lambda, 0, sigma);
  UpdateGradient(
      s, rrt, function.DenseA(), function.DenseB(),
      lambda, function.NumSparseConstraints(), sigma);

  gradient = 2 * s * coordinates;
}

}; // namespace optimization
}; // namespace mlpack
