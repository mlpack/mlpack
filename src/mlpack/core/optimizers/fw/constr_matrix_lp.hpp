/**
 * @file constr_matrix_lp.hpp
 * @author Chenzhe Diao
 *
 * Matrix unit Lp ball constrained for FrankWolfe algorithm. The norm is also
 * known as Schatten Matrix Norms, which is the lp norm of the vector of
 * singular values of the matrix. Used as LinearConstrSolverType in FrankWolfe
 * Algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_CONSTR_MATRIX_LP_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_CONSTR_MATRIX_LP_HPP

#include <mlpack/prereqs.hpp>
#include "constr_lpball.hpp"

namespace mlpack {
namespace optimization {
 
/**
 * LinearConstrSolver for FrankWolfe algorithm. The constraint is in matrix
 * atom domain, that is defined by unit ball under Schatten Matrix Norms.
 *
 * Since the p-Schatten Matrix Norm is just the lp norm of the singular value
 * vector, we add a ConstrLpBallSolver (which is lp ball constraint for vector
 * case) inside the class.
 *
 */
class ConstrMatrixLpBallSolver
{
 public:
  /**
   * Construct the solver of constrained linear problem.
   * The constrained domain is the matrix unit lp ball under Schatten Matrix
   * p norm.
   *
   * @param p The constraint is unit lp ball.
   */
  ConstrMatrixLpBallSolver(const double p):
      p(p), vector_lp_solver(p)
  {/* Nothing to do. */}
 
  //! Get the p-norm.
  double P() const { return p; }
  //! Modify the p-norm.
  double& P() { return p;}

  /**
   * Optimizer of Linear Constrained Problem for FrankWolfe.
   *
   * @param X Input local gradient, which is a matrix here.
   * @param S Output optimal solution matrix in the constrained domain
   *          (matrix lp ball).
   */
  void Optimize(const arma::mat& X, arma::mat& S)
  {
    // left singular vectors.
    arma::mat U;
    // right singular vectors.
    arma::mat V;
    // singular value vector.
    arma::vec SingularVal;

    // Solve for SVD.
    if (!svd_econ(U, SingularVal, V, X))
      Log::Fatal << "ConstrMatrixLpBallSolver: armadillo svd_econ() failed!";

    // Change the singular value vector to its optimal dual,
    // to get the optimal dual matrix.
    arma::vec SingularValDual;
    vector_lp_solver.Optimize(SingularVal, SingularValDual);

    S = U * diagmat(SingularValDual) * V.t();
  }

 private:
  //! lp norm, 1<=p<=inf;
  //! use std::numeric_limits<double>::infinity() for inf norm.
  double p;

  //! Solver for vector lp ball optimal dual.
  ConstrLpBallSolver vector_lp_solver;
};

}  // namespace optimization
}  // namespace mlpack

#endif
