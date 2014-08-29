/**
 * @file lin_alg.cpp
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 *
 * This file is part of MLPACK 1.0.10.
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
#include "lin_alg.hpp"
#include <mlpack/core.hpp>

using namespace mlpack;
using namespace math;

/**
 * Auxiliary function to raise vector elements to a specific power.  The sign
 * is ignored in the power operation and then re-added.  Useful for
 * eigenvalues.
 */
void mlpack::math::VectorPower(arma::vec& vec, const double power)
{
  for (size_t i = 0; i < vec.n_elem; i++)
  {
    if (std::abs(vec(i)) > 1e-12)
      vec(i) = (vec(i) > 0) ? std::pow(vec(i), (double) power) :
          -std::pow(-vec(i), (double) power);
    else
      vec(i) = 0;
  }
}

/**
 * Creates a centered matrix, where centering is done by subtracting
 * the sum over the columns (a column vector) from each column of the matrix.
 *
 * @param x Input matrix
 * @param xCentered Matrix to write centered output into
 */
void mlpack::math::Center(const arma::mat& x, arma::mat& xCentered)
{
  // Get the mean of the elements in each row.
  arma::vec rowMean = arma::sum(x, 1) / x.n_cols;

  xCentered = x - arma::repmat(rowMean, 1, x.n_cols);
}

/**
 * Whitens a matrix using the singular value decomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is the identity
 * matrix.
 */
void mlpack::math::WhitenUsingSVD(const arma::mat& x,
                                  arma::mat& xWhitened,
                                  arma::mat& whiteningMatrix)
{
  arma::mat covX, u, v, invSMatrix, temp1;
  arma::vec sVector;

  covX = ccov(x);

  svd(u, sVector, v, covX);

  size_t d = sVector.n_elem;
  invSMatrix.zeros(d, d);
  invSMatrix.diag() = 1 / sqrt(sVector);

  whiteningMatrix = v * invSMatrix * trans(u);

  xWhitened = whiteningMatrix * x;
}

/**
 * Whitens a matrix using the eigendecomposition of the covariance matrix.
 * Whitening means the covariance matrix of the result is the identity matrix.
 */
void mlpack::math::WhitenUsingEig(const arma::mat& x,
                                  arma::mat& xWhitened,
                                  arma::mat& whiteningMatrix)
{
  arma::mat diag, eigenvectors;
  arma::vec eigenvalues;

  // Get eigenvectors of covariance of input matrix.
  eig_sym(eigenvalues, eigenvectors, ccov(x));

  // Generate diagonal matrix using 1 / sqrt(eigenvalues) for each value.
  VectorPower(eigenvalues, -0.5);
  diag.zeros(eigenvalues.n_elem, eigenvalues.n_elem);
  diag.diag() = eigenvalues;

  // Our whitening matrix is diag(1 / sqrt(eigenvectors)) * eigenvalues.
  whiteningMatrix = diag * trans(eigenvectors);

  // Now apply the whitening matrix.
  xWhitened = whiteningMatrix * x;
}

/**
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N.
 */
void mlpack::math::RandVector(arma::vec& v)
{
  v.zeros();

  for (size_t i = 0; i + 1 < v.n_elem; i += 2)
  {
    double a = Random();
    double b = Random();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i]     = first_term * cos(second_term);
    v[i + 1] = first_term * sin(second_term);
  }

  if ((v.n_elem % 2) == 1)
  {
    v[v.n_elem - 1] = sqrt(-2 * log(math::Random())) * cos(2 * M_PI *
        math::Random());
  }

  v /= sqrt(dot(v, v));
}

/**
 * Orthogonalize x and return the result in W, using eigendecomposition.
 * We will be using the formula \f$ W = x (x^T x)^{-0.5} \f$.
 */
void mlpack::math::Orthogonalize(const arma::mat& x, arma::mat& W)
{
  // For a matrix A, A^N = V * D^N * V', where VDV' is the
  // eigendecomposition of the matrix A.
  arma::mat eigenvalues, eigenvectors;
  arma::vec egval;
  eig_sym(egval, eigenvectors, ccov(x));
  VectorPower(egval, -0.5);

  eigenvalues.zeros(egval.n_elem, egval.n_elem);
  eigenvalues.diag() = egval;

  arma::mat at = (eigenvectors * eigenvalues * trans(eigenvectors));

  W = at * x;
}

/**
 * Orthogonalize x in-place.  This could be sped up by a custom
 * implementation.
 */
void mlpack::math::Orthogonalize(arma::mat& x)
{
  Orthogonalize(x, x);
}

/**
 * Remove a certain set of rows in a matrix while copying to a second matrix.
 *
 * @param input Input matrix to copy.
 * @param rowsToRemove Vector containing indices of rows to be removed.
 * @param output Matrix to copy non-removed rows into.
 */
void mlpack::math::RemoveRows(const arma::mat& input,
                              const std::vector<size_t>& rowsToRemove,
                              arma::mat& output)
{
  const size_t nRemove = rowsToRemove.size();
  const size_t nKeep = input.n_rows - nRemove;

  if (nRemove == 0)
  {
    output = input; // Copy everything.
  }
  else
  {
    output.set_size(nKeep, input.n_cols);

    size_t curRow = 0;
    size_t removeInd = 0;
    // First, check 0 to first row to remove.
    if (rowsToRemove[0] > 0)
    {
      // Note that this implies that n_rows > 1.
      output.rows(0, rowsToRemove[0] - 1) = input.rows(0, rowsToRemove[0] - 1);
      curRow += rowsToRemove[0];
    }

    // Now, check i'th row to remove to (i + 1)'th row to remove, until i is the
    // penultimate row.
    while (removeInd < nRemove - 1)
    {
      const size_t height = rowsToRemove[removeInd + 1] -
          rowsToRemove[removeInd] - 1;

      if (height > 0)
      {
        output.rows(curRow, curRow + height - 1) =
            input.rows(rowsToRemove[removeInd] + 1,
                       rowsToRemove[removeInd + 1] - 1);
        curRow += height;
      }

      removeInd++;
    }

    // Now that i is the last row to remove, check last row to remove to last
    // row.
    if (rowsToRemove[removeInd] < input.n_rows - 1)
    {
      output.rows(curRow, nKeep - 1) = input.rows(rowsToRemove[removeInd] + 1,
          input.n_rows - 1);
    }
  }
}
