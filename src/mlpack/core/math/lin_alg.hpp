/**
 * @file lin_alg.hpp
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 */

#ifndef __MLPACK_CORE_MATH_LIN_ALG_HPP
#define __MLPACK_CORE_MATH_LIN_ALG_HPP

#include <mlpack/core.hpp>

#define max_rand_i 100000

/**
 * Linear algebra utilities.
 *
 * This includes, among other things, Map, Sum, Addition, Subtraction,
 * Multiplication, Hadamard product (entry-wise multiplication), Whitening,
 * Random vectors on the unit sphere, Random uniform matrices, Random
 * normal matrices, creating a Submatrix that is a slice of selected columns of
 * a matrix, Block matrix construction from a base matrix
 * Note that the __private is temporary until this code is merged into a larger
 * namespace of linear algebra utilities
 */
namespace mlpack {
namespace math {

/**
 * Auxiliary function to raise vector elements to a specific power.  The sign
 * is ignored in the power operation and then re-added.  Useful for
 * eigenvalues.
 */
void VectorPower(arma::vec& vec, double power)
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
void Center(const arma::mat& x, arma::mat& xCentered)
{
  // Sum matrix along dimension 0 (that is, sum elements in each row).
  arma::vec rowVectorSum = arma::sum(x, 1);
  rowVectorSum /= x.n_cols; // scale

  xCentered.set_size(x.n_rows, x.n_cols);
  for (size_t i = 0; i < x.n_rows; i++)
    xCentered.row(i) = x.row(i) - rowVectorSum(i);
}

/**
 * Whitens a matrix using the singular value decomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is
 * the identity matrix
 */
void WhitenUsingSVD(const arma::mat& x,
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
 * Whitens a matrix using the eigendecomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is
 * the identity matrix
 */
void WhitenUsingEig(const arma::mat& x,
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
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N
 */
void RandVector(arma::vec &v)
{
  v.zeros();

  for (size_t i = 0; i + 1 < v.n_elem; i += 2)
  {
    double a = drand48();
    double b = drand48();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i]     = first_term * cos(second_term);
    v[i + 1] = first_term * sin(second_term);
  }

  if ((v.n_elem % 2) == 1)
  {
    v[v.n_elem - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  }

  v /= sqrt(dot(v, v));
}

/**
 * Orthogonalize x and return the result in W, using eigendecomposition.
 * We will be using the formula \f$ W = x (x^T x)^{-0.5} \f$.
 */
void Orthogonalize(const arma::mat& x, arma::mat& W)
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
void Orthogonalize(arma::mat& x)
{
  Orthogonalize(x, x);
}

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_METHODS_FASTICA_LIN_ALG_HPP
