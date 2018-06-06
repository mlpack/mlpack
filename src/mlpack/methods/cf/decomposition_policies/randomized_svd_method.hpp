/**
 * @file randomized_svd_method.hpp
 * @author Haritha Nair
 *
 * Implementation of the randomized svd method for use in
 * Collaborative Fitlering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_RANDOMIZED_SVD_METHOD_HPP
#define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_RANDOMIZED_SVD_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/randomized_svd/randomized_svd.hpp>

namespace mlpack {
namespace cf {

/**
 * Implementation of the Randomized SVD policy to act as a wrapper when
 * accessing Randomized SVD from within CFType.
 */
class RandomizedSVDPolicy
{
 public:
  /**
   * Use randomized SVD method to perform collaborative filtering
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
   * Apply Collaborative Filtering to the provided data set using the
   * randomized SVD.
   *
   * @param data Data matrix: dense matrix (coordinate lists) 
   *    or sparse matrix(cleaned).
   * @param cleanedData item user table in form of sparse matrix.
   * @param rank Rank parameter for matrix factorization.
   * @param w First matrix formed after decomposition.
   * @param h Second matrix formed after decomposition.
   * @param maxIterations Maximum number of iterations.
   * @param minResidue Residue required to terminate.
   * @param mit Whether to terminate only when maxIterations is reached.
   */
  template<typename MatType>
  void Apply(const MatType& /* data */,
             const arma::sp_mat& cleanedData,
             const size_t rank,
             arma::mat& w,
             arma::mat& h,
             const size_t maxIterations,
             const double /* minResidue */,
             const bool /* mit */)
  {
    arma::vec sigma;

    // Do singular value decomposition using the randomized SVD algorithm.
    svd::RandomizedSVD rsvd(iteratedPower, maxIterations);
    rsvd.Apply(cleanedData, w, sigma, h, rank);

    // Sigma matrix is multiplied to w.
    w = w * arma::diagmat(sigma);

    // Take transpose of the matrix h as required by CF class.
    h = arma::trans(h);
  }

  //! Get the size of the normalized power iterations.
  size_t IteratedPower() const { return iteratedPower; }
  //! Modify the size of the normalized power iterations.
  size_t& IteratedPower() { return iteratedPower; }

  //! Get the number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the number of iterations.
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! Locally stored size of the normalized power iterations.
  size_t iteratedPower;

  //! Locally stored number of iterations.
  size_t maxIterations;
};

} // namespace cf
} // namespace mlpack

#endif
