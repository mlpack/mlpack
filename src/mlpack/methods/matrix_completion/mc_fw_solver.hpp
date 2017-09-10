/**
 * @file mc_fw_solver.hpp
 * @author Chenzhe Diao
 *
 * A thin wrapper to use Frank-Wolfe type optimizer to solve low rank matrix
 * completion problems.
 *
 * Matrix Schatten p-norm is just the lp norm of the matrix singular
 * value vector.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_SOLVER_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_SOLVER_HPP

#include <mlpack/core/optimizers/fw/frank_wolfe.hpp>
#include <mlpack/core/optimizers/fw/constr_matrix_lp.hpp>
#include "mc_fw_function.hpp"
#include "mc_fw_update_matrix.hpp"

namespace mlpack {
namespace matrix_completion {

/**
 * This class implements the Frank-Wolfe type algorithm for solving
 * matrix completion problems. That is, given known values M_ij's, the
 * following optimization problem is solved to fill in the remaining unknown
 * values of X:
 *
 * \f[
 *   min 1/2 \sum_{(i,j)\in \Omega} (X_ij - M_ij)^2 \qquad s.t.~ ||X||_* <= tau.
 * \f]
 *
 * where ||X||_* denotes the nuclear norm (sum of singular values of X).
 *
 * This type of optimizer needs some prior knowledge of nuclear norm estimate
 * tau, which is input as constraint. All the constructors without tau are not
 * applicable for this solver.
 *
 * To solve this optimization problem, a Frank-Wolfe (Conditional Gradient) type
 * algorithm is used. See
 *
 * @code
 * Rao N, Shah P, Wright S
 * Forward--backward greedy algorithms for atomic norm regularization.
 *
 * IEEE Transactions on Signal Processing 63(21):5798–5811
 * @endcode
 *
 */
class MCFWSolver
{
 public:
  //! Constraint solver to find a new atom for matrix completion problem.
  //! Schatten 1-norm is just the nuclear norm.
  using ConstraintSolver = optimization::ConstrMatrixLpBallSolver;

  /**
   * Construct a matrix completion problem solver, specifying the initial point
   * of the optimization.
   *
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   * @param tau Nuclear norm in constraint.
   */
  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values,
             const double tau) :
      function(indices, values, m, n),
      fwSolver(ConstraintSolver(1), UpdateMatrix(tau), 50000)
  { /* Nothing to do. */ }

  //! This constructor type is not supported.
  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values,
             const size_t r) :
      function(indices, values, m, n),
      fwSolver(ConstraintSolver(1), UpdateMatrix(0))
  { Log::Fatal << "No such constructor!" << std::endl; }

  //! This constructor type is not supported.
  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values,
             const arma::mat& initialPoint) :
      function(indices, values, m, n),
      fwSolver(ConstraintSolver(1), UpdateMatrix(0))
  { Log::Fatal << "No such constructor!" << std::endl; }

  //! This constructor type is not supported.
  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values) :
      function(indices, values, m, n),
      fwSolver(ConstraintSolver(1), UpdateMatrix(0))
  { Log::Fatal << "No such constructor!" << std::endl; }

  /**
   * Recover the low-rank matrix.
   *
   * @param recovered Matrix to save the output of the recovered matrix.
   */
  void Recover(arma::mat& recovered, const size_t m, const size_t n)
  {
    fwSolver.Optimize(function, recovered);
  }

 private:
  //! Function to be optimized.
  MatrixCompletionFWFunction function;

  //! Frank Wolfe type optimization solver.
  optimization::FrankWolfe<ConstraintSolver, UpdateMatrix> fwSolver;
};
} // namespace matrix_completion
} // namespace mlpack
#endif
