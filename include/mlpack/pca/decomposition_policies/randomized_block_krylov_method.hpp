/**
 * @file methods/pca/decomposition_policies/randomized_block_krylov_method.hpp
 * @author Marcus Edel
 *
 * Implementation of the randomized block krylov SVD method for use in the
 * Principal Components Analysis method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_BLOCK_KRYLOV_HPP
#define MLPACK_METHODS_PCA_DECOMPOSITION_POLICIES_RANDOMIZED_BLOCK_KRYLOV_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/block_krylov_svd/randomized_block_krylov_svd.hpp>

namespace mlpack {

/**
 * Implementation of the randomized block krylov SVD policy.
 */
class RandomizedBlockKrylovSVDPolicy
{
 public:
  /**
   * Use randomized block krylov SVD method to perform the principal components
   * analysis (PCA).
   *
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   * @param blockSize The block size, must be >= rank (Default: rank + 10).
   */
  RandomizedBlockKrylovSVDPolicy(const size_t maxIterations = 2,
                                 const size_t blockSize = 0) :
      maxIterations(maxIterations),
      blockSize(blockSize)
  {
    /* Nothing to do here */
  }

  /**
   * Apply Principal Component Analysis to the provided data set using the
   * randomized block krylov SVD method.
   *
   * @param data Data matrix.
   * @param centeredData Centered data matrix.
   * @param transformedData Matrix to put results of PCA into.
   * @param eigVal Vector to put eigenvalues into.
   * @param eigvec Matrix to put eigenvectors (loadings) into.
   * @param rank Rank of the decomposition.
   */
  template<typename InMatType, typename MatType, typename VecType>
  void Apply(const InMatType& /* data */,
             const MatType& centeredData,
             MatType& transformedData,
             VecType& eigVal,
             MatType& eigvec,
             const size_t rank)
  {
    // This matrix will store the right singular vectors; we do not need them.
    MatType v;

    // Do singular value decomposition using the randomized block krylov SVD
    // algorithm.
    RandomizedBlockKrylovSVD rsvd(maxIterations, blockSize);
    rsvd.Apply(centeredData, eigvec, eigVal, v, rank);

    // Now we must square the singular values to get the eigenvalues.
    // In addition we must divide by the number of points, because the
    // covariance matrix is X * X' / (N - 1).
    eigVal %= eigVal / (centeredData.n_cols - 1);

    // Project the samples to the principals.
    transformedData = trans(eigvec) * centeredData;
  }

  //! Get the number of iterations for the power method.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the number of iterations for the power method.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the block size.
  size_t BlockSize() const { return blockSize; }
  //! Modify the block size.
  size_t& BlockSize() { return blockSize; }

 private:
  //! Locally stored number of iterations for the power method.
  size_t maxIterations;

  //! Locally stored block size value.
  size_t blockSize;
};

} // namespace mlpack

#endif
