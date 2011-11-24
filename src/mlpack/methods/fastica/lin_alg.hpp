/**
 * @file lin_alg.h
 * @author Nishant Mehta
 *
 * Linear algebra utilities.
 */

#ifndef __MLPACK_METHODS_FASTICA_LIN_ALG_HPP
#define __MLPACK_METHODS_FASTICA_LIN_ALG_HPP

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
namespace fastica {
namespace linalg__private {

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
 * @param X Input matrix
 * @param X_centered Matrix to write centered output into
 */
void Center(const arma::mat& X, arma::mat& X_centered)
{
  // Sum matrix along dimension 0 (that is, sum elements in each row).
  arma::vec row_vector_sum = arma::sum(X, 1);
  row_vector_sum /= X.n_cols; // scale

  X_centered.set_size(X.n_rows, X.n_cols);
  for (size_t i = 0; i < X.n_rows; i++)
    X_centered.row(i) = X.row(i) - row_vector_sum(i);
}

/**
 * Whitens a matrix using the singular value decomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is
 * the identity matrix
 */
void WhitenUsingSVD(const arma::mat& X,
                    arma::mat& X_whitened,
                    arma::mat& whitening_matrix)
{
  arma::mat cov_X, U, V, inv_S_matrix, temp1;
  arma::vec S_vector;

  cov_X = ccov(X);

  svd(U, S_vector, V, cov_X);

  size_t d = S_vector.n_elem;
  inv_S_matrix.zeros(d, d);
  inv_S_matrix.diag() = 1 / sqrt(S_vector);

  whitening_matrix = V * inv_S_matrix * trans(U);

  X_whitened = whitening_matrix * X;
}

/**
 * Whitens a matrix using the eigendecomposition of the covariance
 * matrix. Whitening means the covariance matrix of the result is
 * the identity matrix
 */
void WhitenUsingEig(const arma::mat& X,
                    arma::mat& X_whitened,
                    arma::mat& whitening_matrix)
{
  arma::mat diag, eigenvectors;
  arma::vec eigenvalues;

  // Get eigenvectors of covariance of input matrix.
  eig_sym(eigenvalues, eigenvectors, ccov(X));

  // Generate diagonal matrix using 1 / sqrt(eigenvalues) for each value.
  VectorPower(eigenvalues, -0.5);
  diag.zeros(eigenvalues.n_elem, eigenvalues.n_elem);
  diag.diag() = eigenvalues;

  // Our whitening matrix is diag(1 / sqrt(eigenvectors)) * eigenvalues.
  whitening_matrix = diag * trans(eigenvectors);

  // Now apply the whitening matrix.
  X_whitened = whitening_matrix * X;
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

  if((v.n_elem % 2) == 1)
  {
    v[v.n_elem - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  }

  v /= sqrt(dot(v, v));
}

/**
 * Orthogonalize X and return the result in W, using eigendecomposition.
 * We will be using the formula \f$ W = X (X^T X)^{-0.5} \f$.
 */
void Orthogonalize(const arma::mat& X, arma::mat& W)
{
  // For a matrix A, A^N = V * D^N * V', where VDV' is the
  // eigendecomposition of the matrix A.
  arma::mat eigenvalues, eigenvectors;
  arma::vec egval;
  eig_sym(egval, eigenvectors, ccov(X));
  VectorPower(egval, -0.5);

  eigenvalues.zeros(egval.n_elem, egval.n_elem);
  eigenvalues.diag() = egval;

  arma::mat at = (eigenvectors * eigenvalues * trans(eigenvectors));

  W = at * X;
}

/**
 * Orthogonalize X in-place.  This could be sped up by a custom
 * implementation.
 */
void Orthogonalize(arma::mat& X)
{
  Orthogonalize(X, X);
}

}; // namespace linalg__private
}; // fastica
}; // mlpack

#endif // __MLPACK_METHODS_FASTICA_LIN_ALG_HPP
