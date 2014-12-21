/**
 * @file lin_alg.hpp
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_MATH_LIN_ALG_HPP
#define __MLPACK_CORE_MATH_LIN_ALG_HPP

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

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_LIN_ALG_HPP
