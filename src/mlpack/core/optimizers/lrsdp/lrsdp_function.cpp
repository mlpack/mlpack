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
    c_sparse(initialPoint.n_rows, initialPoint.n_rows),
    c_dense(initialPoint.n_rows, initialPoint.n_rows, arma::fill::zeros),
    hasModifiedSparseObjective(false),
    hasModifiedDenseObjective(false),
    a_sparse(numSparseConstraints),
    b_sparse(numSparseConstraints),
    a_dense(numDenseConstraints),
    b_dense(numDenseConstraints),
    initialPoint(initialPoint)
{
  if (initialPoint.n_rows < initialPoint.n_cols)
    throw invalid_argument("initialPoint n_cols > n_rows");
}

double LRSDPFunction::Evaluate(const arma::mat& coordinates) const
{
  return -accu(coordinates * trans(coordinates));
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
    return trace(a_sparse[index] * rrt) - b_sparse[index];
  const size_t index1 = index - NumSparseConstraints();
  return trace(a_dense[index1] * rrt) - b_dense[index1];
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
  convert << "  Sparse Constraint b_i values: " << b_sparse.t();
  convert << "  Dense Constraint b_i values: " << b_dense.t();
  return convert.str();
}

template <typename MatrixType>
static inline void
updateObjective(double &objective,
                const arma::mat &rrt,
                const std::vector<MatrixType> &ais,
                const arma::vec &bis,
                const arma::vec &lambda,
                size_t lambda_offset,
                double sigma)
{
  for (size_t i = 0; i < ais.size(); ++i)
  {
    // Take the trace subtracted by the b_i.
    double constraint = trace(ais[i] * rrt) - bis[i];
    objective -= (lambda[lambda_offset + i] * constraint);
    objective += (sigma / 2.) * constraint * constraint;
  }
}

template <typename MatrixType>
static inline void
updateGradient(arma::mat &s,
               const arma::mat &rrt,
               const std::vector<MatrixType> &ais,
               const arma::vec &bis,
               const arma::vec &lambda,
               size_t lambda_offset,
               double sigma)
{
  for (size_t i = 0; i < ais.size(); ++i)
  {
    const double constraint = trace(ais[i] * rrt) - bis[i];
    const double y = lambda[lambda_offset + i] - sigma * constraint;
    s -= y * ais[i];
  }
}

namespace mlpack {
namespace optimization {

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
    objective += trace(function.C_sparse() * rrt);
  if (function.hasDenseObjective())
    objective += trace(function.C_dense() * rrt);

  // Now each constraint.
  updateObjective(
      objective, rrt, function.A_sparse(), function.B_sparse(),
      lambda, 0, sigma);
  updateObjective(
      objective, rrt, function.A_dense(), function.B_dense(),
      lambda, function.NumSparseConstraints(), sigma);

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
  arma::mat s(function.n(), function.n(), arma::fill::zeros);

  if (function.hasSparseObjective())
    s += function.C_sparse();
  if (function.hasDenseObjective())
    s += function.C_dense();

  updateGradient(
      s, rrt, function.A_sparse(), function.B_sparse(),
      lambda, 0, sigma);
  updateGradient(
      s, rrt, function.A_dense(), function.B_dense(),
      lambda, function.NumSparseConstraints(), sigma);

  gradient = 2 * s * coordinates;
}

}; // namespace optimization
}; // namespace mlpack
