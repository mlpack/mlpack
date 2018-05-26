/**
 * @file regularized_svd_method.hpp
 * @author Haritha Nair
 *
 * Implementation of the regularized svd method for use in the Collaborative
 * Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_REGULARIZED_SVD_METHOD_HPP
#define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_REGULARIZED_SVD_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/regularized_svd/regularized_svd.hpp>

namespace mlpack {
namespace cf {

/**
 * Implementation of the Regularized SVD policy to act as a wrapper when
 * accessing Regularized SVD from within CFType.
 */
class RegSVDPolicy
{
 public:
  /**
   * Use regularized SVD method to perform collaborative filtering.
   *
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   */
  RegSVDPolicy(const size_t maxIterations = 10) :
      maxIterations(maxIterations)
  {
    /* Nothing to do here */
  }

  /**
   * Apply Collaborative Filtering to the provided data set using the
   * regularized SVD.
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
  void Apply(const arma::mat& data,
             const arma::sp_mat& /* cleanedData */,
             const size_t rank,
             arma::mat& w,
             arma::mat& h,
             const size_t maxIterations,
             const double /* minResidue */,
             const bool /* mit */)
  {
    // Do singular value decomposition using the regularized SVD algorithm.
    svd::RegularizedSVD<> regsvd(maxIterations);
    regsvd.Apply(data, rank, w, h);
  }

  //! Get the number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the number of iterations.
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! Locally stored number of iterations.
  size_t maxIterations;
};

} // namespace cf
} // namespace mlpack

#endif
