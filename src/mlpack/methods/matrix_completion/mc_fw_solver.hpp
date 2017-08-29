/**
 * @file mc_fw_solver.hpp
 * @author Chenzhe Diao
 *
 * A thin wrapper around Matrix Schatten p-norm minimization to solve
 * low rank matrix completion problems using FrankWolfe type solver.
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
class MCFWSolver
{
 public:

 /*
  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values,
             const size_t r) :
    function(indices, values, m, n), r(r)
    {
      ConstrMatrixLpBallSolver constrSolver(1);
      optimization::UpdateMatrix updateRule;

      fwSolver = optimization::FrankWolfe<
          optimization::ConstrMatrixLpBallSolver, UpdateMatrix>(
          constrSolver, updateRule);
    }

  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values,
             const arma::mat& initialPoint) :
    function(indices, values, m, n, initialPoint)
    {
      ConstrMatrixLpBallSolver constrSolver(1);
      optimization::UpdateMatrix updateRule;
        
      fwSolver = optimization::FrankWolfe<
          optimization::ConstrMatrixLpBallSolver, UpdateMatrix>(
          constrSolver, updateRule);
    }
*/

  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values) :
      function(indices, values, m, n),
      fwSolver(optimization::ConstrMatrixLpBallSolver(1),
        UpdateMatrix(1))
  { Log::Fatal << "No such constructor!" << std::endl;  }

  MCFWSolver(const size_t m,
             const size_t n,
             const arma::umat& indices,
             const arma::vec& values,
             const double tau) :
    function(indices, values, m, n),
    fwSolver(optimization::ConstrMatrixLpBallSolver(1),
        UpdateMatrix(tau), 1000)
    {}

  void Recover(arma::mat& recovered, const size_t m, const size_t n)
  {
    fwSolver.Optimize(function, recovered);
  }

 private:
  //! Function to be optimized.
  MatrixCompletionFWFunction function;

  optimization::FrankWolfe<optimization::ConstrMatrixLpBallSolver,
      UpdateMatrix> fwSolver;
};
} // namespace matrix_completion
} // namespace mlpack
#endif
