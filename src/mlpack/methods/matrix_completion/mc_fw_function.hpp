/**
 * @file mc_fw_function.hpp
 * @author Chenzhe Diao
 *
 * Optimization object function for matrix completion problem.
 *
 *\f[
 * f(X) = 0.5* \sum_{(i,j)\in \Omega} |X_{i,j}-M_{i,j}|^2
 *\f]
 *
 * Used in FrankWolfe Type solver.
 *
 * Matrix Schatten p-norm is just the lp norm of the matrix singular
 * value vector.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_FUNCTION_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_FUNCTION_HPP

namespace mlpack {
namespace matrix_completion {
class MatrixCompletionFWFunction {
 public:
  MatrixCompletionFWFunction(const arma::umat& indices,
                             const arma::vec& values,
                             const size_t m,
                             const size_t n) :
      indices(indices), values(values), m(m), n(n),
      initialPoint(m, n, arma::fill::zeros)
  { /* Nothing to do. */ }

  MatrixCompletionFWFunction(const arma::umat& indices,
                             const arma::vec& values,
                             const size_t m,
                             const size_t n,
                             const arma::mat initialPoint) :
      indices(indices), values(values), m(m), n(n),
      initialPoint(initialPoint)
  { /* Nothing to do. */ }

    
  double Evaluate(const arma::mat& X)
  {
    double f = 0;
    for (arma::uword i = 0; i<indices.n_cols; i++) {
      arma::uword rind = indices(0, i);
      arma::uword cind = indices(1, i);
      f += std::pow(X(rind, cind) - values(i), 2);
    }
    return 0.5*f;
  }
  
  void Gradient(const arma::mat& X, arma::mat& gradient)
  {
    arma::vec gradientVal = -values;
    for (arma::uword i = 0; i<indices.n_cols; i++) {
      arma::uword rind = indices(0, i);
      arma::uword cind = indices(1, i);
      gradientVal(i) += X(rind, cind);
    }
    
    arma::sp_mat spGradient(indices, gradientVal);
    gradient = arma::mat(spGradient);
  }

private:
  //! Indices for sparse matrix.
  arma::umat indices;
  //! Values for sparse matrix.
  arma::vec values;
  //! Number of rows of the matrix.
  size_t m;
  //! Number of columns of the matrix.
  size_t n;
  //! Initial point of iterations.
  arma::mat initialPoint;
};
}  // namespace matrix_completion
}  // namespace mlpack

#endif
