/**
 * @file mc_sgd_solver.hpp
 * @author Chenzhe Diao
 *
 *
 * A thin wrapper to solve low rank matrix completion problems using SGD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_SGD_SOLVER_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_SGD_SOLVER_HPP

namespace mlpack {
namespace matrix_completion {

class MCSGDSolver {
 public:
  MCSGDSolver(const size_t m,
              const size_t n,
              const arma::umat& indices,
              const arma::vec& values,
              const size_t r) :
      function(m, n, indices, values, r),
      sgd(function),
      leftMat(arma::randu<arma::mat>(m, r)),
      rightMat(arma::randu<arma::mat>(n, r))
  { /* Nothing to do.  */ }
  
  void Recover(arma::mat& recovered, const size_t m, const size_t n)
  {
    arma::mat iterate = arma::join_cols(leftMat, rightMat);
    sgd.Optimize(iterate);
    
    leftMat = iterate.head_rows(m);
    rightMat = iterate.tail_rows(n);
    recovered = leftMat * rightMat.t();
  }

 private:
  MatrixCompletionSGDFunction function;
  optimization::SGD<MatrixCompletionSGDFunction> sgd;
  
  arma::mat leftMat;
  arma::mat rightMat;

};
}
}


#endif
