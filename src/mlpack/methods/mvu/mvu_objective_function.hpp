/**
 * @file mvu_objective_function.hpp
 * @author Ryan Curtin
 *
 * This file defines the standard MVU objective function, implemented to fulfill
 * the "LagrangianFunction" policy defined in
 * fastlib/optimization/aug_lagrangian/aug_lagrangian.h.
 *
 * We use the change of variables suggested by Burer and Monteiro [1] to
 * reformulate the MVU SDP originally provided by Weinberger and Saul [2].
 *
 * [1] S. Burer and R.D.C. Monteiro.  "A nonlinear programming algorithm for
 *     solving semidefinite programs via low-rank factorization," Mathematical
 *     Programming, vol. 95, no. 2, pp. 329-357, 2002.
 *
 * [2] K.Q. Weinberger and L.K. Saul, "An introduction to Nonlinear
 *     Dimensionality Reduction by Maximum Variance Unfolding," Proceedings of
 *     the Twenty-First National Conference on Artificial Intelligence
 *     (AAAI-06), 2006.
 */
#ifndef __MLPACK_METHODS_MVU_MVU_OBJECTIVE_FUNCTION_HPP
#define __MLPACK_METHODS_MVU_MVU_OBJECTIVE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace mvu {

/**
 * The MVU objective function.  This is a reformulation of the SDP proposed
 * originally by Weinberger and Saul.  Their original formulation was:
 *
 * max (trace(K)) subject to:
 *  (1) sum K_ij = 0
 *  (2) K_ii - 2 K_ij + K_jj = || x_i - x_j ||^2 ; for all (i, j) nearest
 *                                                 neighbors
 *  (3) K >= 0 (K is positive semidefinite)
 *
 * We reformulate, taking K = R R^T.  This gives:
 *
 * max (R * R^T) subject to
 *  (1) sum (R * R^T) = 0
 *  (2) (R R^T)_ii - 2 (R R^T)_ij + (R R^T)_jj = || x_i - x_j ||^2 ;
 *        for all (i, j) nearest neighbors
 *
 * Now, our optimization problem is easier.  The total number of constraints is
 * equal to the number of points multiplied by the number of nearest neighbors.
 */
class MVUObjectiveFunction
{
 public:
  MVUObjectiveFunction();
  MVUObjectiveFunction(const arma::mat& initial_point,
                       const size_t newDim,
                       const size_t numNeighbors);

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  size_t NumConstraints() const { return numNeighbors * initialPoint.n_cols; }

  double EvaluateConstraint(const size_t index, const arma::mat& coordinates);
  void GradientConstraint(const size_t index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint() const { return initialPoint; }

 private:
  arma::mat initialPoint;
  size_t numNeighbors;

  // These hold the output of the nearest neighbors computation (done in the
  // constructor).
  arma::Mat<size_t> neighborIndices;
  arma::mat neighborDistances;
};

}; // namespace mvu
}; // namespace mlpack

#endif
