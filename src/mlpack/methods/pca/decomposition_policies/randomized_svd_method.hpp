/**
 * @file randomized_svd_method.hpp
 * @author Marcus Edel
 *
 * Implementation of the randomized svd method for use in the Principal
 * Components Analysis method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_SVD_METHOD_HPP
#define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_SVD_METHOD_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/randomized_svd/randomized_svd.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

namespace mlpack {
namespace pca {

/**
 * Implementation of the randomized SVD policy.
 */
class RandomizedSVDPolicy
{
  public:
  /**
   * Use randomized SVD method to perform the principal components analysis
   * (PCA).
   *
   * @param iteratedPower Size of the normalized power iterations
   *        (Default: rank + 2).
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   */
  RandomizedSVDPolicy(const size_t iteratedPower = 0,
                      const size_t maxIterations = 2) :
      iteratedPower(iteratedPower),
      maxIterations(maxIterations)
  {
    /* Nothing to do here */
  }

  /**
   * Apply Principal Component Analysis to the provided data set using the
   * randomized SVD.
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
             const size_t rank)
  {
    // This matrix will store the right singular values; we do not need them.
    arma::mat v;

    // Do singular value decomposition using the randomized SVD algorithm.
    svd::RandomizedSVD rsvd(iteratedPower, maxIterations);
    rsvd.Apply(data, eigvec, eigVal, v, rank);

    // Now we must square the singular values to get the eigenvalues.
    // In addition we must divide by the number of points, because the
    // covariance matrix is X * X' / (N - 1).
    eigVal %= eigVal / (data.n_cols - 1);

    // Project the samples to the principals.
    transformedData = arma::trans(eigvec) * centeredData;
  }

  //! Get the size of the normalized power iterations.
  size_t IteratedPower() const { return iteratedPower; }
  //! Modify the size of the normalized power iterations.
  size_t& IteratedPower() { return iteratedPower; }

  //! Get the number of iterations for the power method.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the number of iterations for the power method.
  size_t& MaxIterations() { return maxIterations; }

  private:
    //! Locally stored size of the normalized power iterations.
    size_t iteratedPower;

    //! Locally stored number of iterations for the power method.
    size_t maxIterations;
};

} // namespace pca
} // namespace mlpack

#endif
