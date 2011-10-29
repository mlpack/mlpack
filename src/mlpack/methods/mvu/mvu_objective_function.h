/***
 * @file mvu_objective_function.h
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

#ifndef __MLPACK_MVU_OBJECTIVE_FUNCTCLIN_H
#define __MLPACK_MVU_OBJECTIVE_FUNCTCLIN_H

#include <mlpack/core.h>

namespace mlpack {
namespace mvu {

/***
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
class MVUObjectiveFunction {
 public:
  MVUObjectiveFunction();
  MVUObjectiveFunction(arma::mat& initial_point);

  double Evaluate(const arma::mat& coordinates);
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);

  int NumConstraints() const { return num_neighbors_ * initial_point_.n_cols; }

  double EvaluateConstraint(int index, const arma::mat& coordinates);
  void GradientConstraint(int index,
                          const arma::mat& coordinates,
                          arma::mat& gradient);

  const arma::mat& GetInitialPoint() { return initial_point_; }

 private:
  arma::mat initial_point_;
  int num_neighbors_;

  // These hold the output of the nearest neighbors computation (done in the
  // constructor).
  arma::Col<index_t> neighbor_indices_;
  arma::vec neighbor_distances_;
};

}; // namespace mvu
}; // namespace mlpack

#endif
