/**
 * @file quic_svd_method.hpp
 * @author Marcus Edel
 *
 * Implementation of the QUIC-SVD policy for use in the Principal Components
 * Analysis method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_QUIC_SVD_METHOD_HPP
#define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_QUIC_SVD_METHOD_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/quic_svd/quic_svd.hpp>

namespace mlpack {
namespace pca {

/**
 * Implementation of the QUIC-SVD policy.
 */
class QUICSVDPolicy
{
  public:

  /**
   * Use QUIC-SVD method to perform the principal components analysis (PCA).
   *
   * @param epsilon Error tolerance fraction for calculated subspace.
   * @param delta Cumulative probability for Monte Carlo error lower bound.
   */
  QUICSVDPolicy(const double epsilon = 0.03, const double delta = 0.1) :
       epsilon(epsilon),
       delta(delta)
  {
    /* Nothing to do here */
  }

  /**
   * Apply Principal Component Analysis to the provided data set using the
   * QUIC-SVD method.
   *
   * @param data Data matrix.
   * @param centeredData Centered data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigVal Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   * @param rank Rank of the decomposition.
   */
  void Apply(const arma::mat& data,
             const arma::mat& centeredData,
             arma::mat& transformedData,
             arma::vec& eigVal,
             arma::mat& eigvec,
             const size_t /* rank */)
  {
    // This matrix will store the right singular values; we do not need them.
    arma::mat v, sigma;

    // Do singular value decomposition using the QUIC-SVD algorithm.
    svd::QUIC_SVD quicsvd(centeredData, eigvec, v, sigma, epsilon, delta);

    // Now we must square the singular values to get the eigenvalues.
    // In addition we must divide by the number of points, because the
    // covariance matrix is X * X' / (N - 1).
    eigVal = arma::pow(arma::diagvec(sigma), 2) / (data.n_cols - 1);

    // Project the samples to the principals.
    transformedData = arma::trans(eigvec) * centeredData;
  }

  //! Get the error tolerance fraction for calculated subspace.
  double Epsilon() const { return epsilon; }
  //! Modify the error tolerance fraction for calculated subspace.
  double& Epsilon() { return epsilon; }

  //! Get the cumulative probability for Monte Carlo error lower bound.
  double Delta() const { return delta; }
  //! Modify the cumulative probability for Monte Carlo error lower bound.
  double& Delta() { return delta; }

  private:
    //! Error tolerance fraction for calculated subspace.
    double epsilon;

    //! Cumulative probability for Monte Carlo error lower bound.
    double delta;
};

} // namespace pca
} // namespace mlpack

#endif
