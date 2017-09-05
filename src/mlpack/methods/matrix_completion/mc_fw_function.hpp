/**
 * @file mc_fw_function.hpp
 * @author Chenzhe Diao
 *
 * Optimization object function for matrix completion problem using Frank-Wolfe
 * type algorithms.
 *
 *\f[
 * f(X) = 0.5* \sum_{(i,j)\in \Omega} |X_{i,j}-M_{i,j}|^2
 *\f]
 *
 * Used in mc_fw_solver.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_FUNCTION_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_FUNCTION_HPP

#include <mlpack/core/optimizers/fw/frank_wolfe.hpp>

namespace mlpack {
namespace matrix_completion {
class MatrixCompletionFWFunction {
 public:
  /**
   * Construct the object function.
   *
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   */
  MatrixCompletionFWFunction(const arma::umat& indices,
                             const arma::vec& values,
                             const size_t m,
                             const size_t n) :
      indices(indices), values(values), m(m), n(n)
  { /* Nothing to do. */ }

  /**
   * Object function.
   *
   * f(X) = 0.5 * \sum_{i,j} (X_ij - M_ij)^2
   *
   * where M_ij is the known matrix entries.
   */
  double Evaluate(const arma::mat& X)
  {
    double f = 0;
    for (arma::uword i = 0; i < indices.n_cols; i++)
    {
      arma::uword rind = indices(0, i);
      arma::uword cind = indices(1, i);
      f += std::pow(X(rind, cind) - values(i), 2);
    }
    return 0.5 * f;
  }

  /**
   * Gradient of the objective function.
   *
   * gradient_ij = X_ij - M_ij,  for ij \in \Omega
   * gradient_ij = 0,   otherwise.
   *
   * @param X input matrix, with size m x n.
   * @param gradient output gradient matrix.
   */
  void Gradient(const arma::mat& X, arma::mat& gradient)
  {
    arma::vec gradientVal = -values;
    for (arma::uword i = 0; i < indices.n_cols; i++)
    {
      arma::uword rind = indices(0, i);
      arma::uword cind = indices(1, i);
      gradientVal(i) += X(rind, cind);
    }

    arma::sp_mat spGradient(indices, gradientVal);
    gradient = arma::mat(spGradient);
  }

  /**
   * Get the values of a given matrix at positions with known "indices".
   *
   * @param X Input given matrix.
   * @param xValues Output entries of the matrix X.
   */
  void GetKnownEntries(const arma::mat& X, arma::vec& xValues)
  {
    xValues.set_size(arma::size(values));
    for (size_t i = 0; i < indices.n_cols; i++)
      xValues(i) = X(indices(0, i), indices(1, i));
  }

  //! Get the known elements.
  const arma::vec& Values() const { return values; }

private:
  //! Indices for sparse matrix.
  arma::umat indices;

  //! Values for sparse matrix.
  arma::vec values;

  //! Number of rows of the matrix.
  size_t m;

  //! Number of columns of the matrix.
  size_t n;
}; // class MatrixCompletionFWFunction
}  // namespace matrix_completion
}  // namespace mlpack

#endif
