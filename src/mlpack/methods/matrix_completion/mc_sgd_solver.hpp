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

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include "mc_sgd_function.hpp"


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
      leftMat(arma::randu<arma::mat>(m, r)),
      rightMat(arma::randu<arma::mat>(n, r))
  { /* Nothing to do.  */ }

    MCSGDSolver(const size_t m,
                const size_t n,
                const arma::umat& indices,
                const arma::vec& values,
                const arma::mat& initialPoint) :
      function(m, n, indices, values, DefaultRank(m, n, indices.n_cols)),
      leftMat(arma::randu<arma::mat>(m, DefaultRank(m, n, indices.n_cols))),
      rightMat(arma::randu<arma::mat>(n, DefaultRank(m, n, indices.n_cols)))
    { /* */ }

    MCSGDSolver(const size_t m,
                const size_t n,
                const arma::umat& indices,
                const arma::vec& values) :
      function(m, n, indices, values, DefaultRank(m, n, indices.n_cols)),
      leftMat(arma::randu<arma::mat>(m, DefaultRank(m, n, indices.n_cols))),
      rightMat(arma::randu<arma::mat>(n, DefaultRank(m, n, indices.n_cols)))
    { /* */ }



  
  void Recover(arma::mat& recovered, const size_t m, const size_t n)
  {
    arma::mat iterate = arma::join_cols(leftMat, rightMat);
    sgd.Optimize(function, iterate);
    
    leftMat = iterate.head_rows(m);
    rightMat = iterate.tail_rows(n);
    recovered = leftMat * rightMat.t();
  }

 private:
  MatrixCompletionSGDFunction function;
  optimization::StandardSGD sgd;
  
  arma::mat leftMat;
  arma::mat rightMat;

  size_t DefaultRank(const size_t m, const size_t n, const size_t p)
  {
    // If r = O(sqrt(p)), then we are guaranteed an exact solution.
    // For more details, see
    //
    //   On the rank of extreme matrices in semidefinite programs and the
    //   multiplicity of optimal eigenvalues.
    //   Pablo Moscato, Michael Norman, and Gabor Pataki.
    //   Math Oper. Res., 23(2). 1998.
    const size_t mpn = m + n;
    float r = 0.5 + sqrt(0.25 + 2 * p);
    if (ceil(r) > mpn)
      r = mpn; // An upper bound on the dimension.
    return ceil(r);
  }

};
}
}


#endif
