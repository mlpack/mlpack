/**
 * @file lin_alg.hpp
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_LIN_ALG_HPP
#define MLPACK_CORE_MATH_LIN_ALG_HPP

#include <mlpack/prereqs.hpp>

/**
 * Linear algebra utility functions, generally performed on matrices or vectors.
 */
namespace mlpack {
namespace math {

/**
 * Auxiliary function to raise vector elements to a specific power.  The sign
 * is ignored in the power operation and then re-added.  Useful for
 * eigenvalues.
 */
void VectorPower(arma::vec& vec, const double power);

/**
 * Creates a centered matrix, where centering is done by subtracting
 * the sum over the columns (a column vector) from each column of the matrix.
 *
 * @param x Input matrix
 * @param xCentered Matrix to write centered output into
 */
void Center(const arma::mat& x, arma::mat& xCentered);

/**
 * Whitens a matrix using the singular value decomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is the identity
 * matrix.
 */
void WhitenUsingSVD(const arma::mat& x,
                    arma::mat& xWhitened,
                    arma::mat& whiteningMatrix);

/**
 * Whitens a matrix using the eigendecomposition of the covariance matrix.
 * Whitening means the covariance matrix of the result is the identity matrix.
 */
void WhitenUsingEig(const arma::mat& x,
                    arma::mat& xWhitened,
                    arma::mat& whiteningMatrix);

/**
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N.
 */
void RandVector(arma::vec& v);

/**
 * Orthogonalize x and return the result in W, using eigendecomposition.
 * We will be using the formula \f$ W = x (x^T x)^{-0.5} \f$.
 */
void Orthogonalize(const arma::mat& x, arma::mat& W);

/**
 * Orthogonalize x in-place.  This could be sped up by a custom
 * implementation.
 */
void Orthogonalize(arma::mat& x);

/**
 * Remove a certain set of rows in a matrix while copying to a second matrix.
 *
 * @param input Input matrix to copy.
 * @param rowsToRemove Vector containing indices of rows to be removed.
 * @param output Matrix to copy non-removed rows into.
 */
void RemoveRows(const arma::mat& input,
                const std::vector<size_t>& rowsToRemove,
                arma::mat& output);

/**
 * Upper triangular representation of a symmetric matrix, scaled such that,
 * dot(Svec(A), Svec(B)) == dot(A, B) for symmetric A, B. Specifically,
 *
 * Svec(K) = [ K_11, sqrt(2) K_12, ..., sqrt(2) K_1n, K_22, ..., sqrt(2) K_2n, ..., K_nn ]^T
 *
 * @param input A symmetric matrix
 * @param output
 */
void Svec(const arma::mat& input, arma::vec& output);

void Svec(const arma::sp_mat& input, arma::sp_vec& output);

/**
 * The inverse of Svec. That is, Smat(Svec(A)) == A.
 *
 * @param input
 * @param output A symmetric matrix
 */
void Smat(const arma::vec& input, arma::mat& output);

/**
 * Return the index such that A[i,j] == factr(i, j) * svec(A)[pos(i, j)],
 * where factr(i, j) = sqrt(2) if i != j and 1 otherwise.
 *
 * @param i
 * @param j
 * @param n
 */
inline size_t SvecIndex(size_t i, size_t j, size_t n);

/**
 * If A is a symmetric matrix, then SymKronId returns an operator Op such that
 *
 *    Op * svec(X) == svec(0.5 * (AX + XA))
 *
 * for every symmetric matrix X
 *
 * @param A
 * @param op
 */
void SymKronId(const arma::mat& A, arma::mat& op);

} // namespace math
} // namespace mlpack

// Partially include implementation
#include "lin_alg_impl.hpp"

#endif // MLPACK_CORE_MATH_LIN_ALG_HPP
