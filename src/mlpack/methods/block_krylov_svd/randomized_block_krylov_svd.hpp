/**
 * @file methods/block_krylov_svd/randomized_block_krylov_svd.hpp
 * @author Marcus Edel
 *
 * An implementation of the randomized block krylov SVD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_BLOCK_KRYLOV_SVD_RANDOMIZED_BLOCK_KRYLOV_SVD_HPP
#define MLPACK_METHODS_BLOCK_KRYLOV_SVD_RANDOMIZED_BLOCK_KRYLOV_SVD_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * Randomized block krylov SVD is a matrix factorization that is based on
 * randomized matrix approximation techniques, developed in in
 * "Randomized Block Krylov Methods for Stronger and Faster Approximate
 * Singular Value Decomposition".
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Musco2015,
 *   author    = {Cameron Musco and Christopher Musco},
 *   title     = {Randomized Block Krylov Methods for Stronger and Faster
 *                Approximate Singular Value Decomposition},
 *   booktitle = {Advances in Neural Information Processing Systems 28: Annual
 *                Conference on Neural Information Processing Systems 2015,
 *                December 7-12, 2015, Montreal, Quebec, Canada},
 *   pages     = {1396--1404},
 *   year      = {2015},
 * }
 * @endcode
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Rating data in the form of coordinate list.
 *
 * const size_t rank = 20; // Rank used for the decomposition.
 *
 * // Make a RandomizedBlockKrylovSVD object.
 * RandomizedBlockKrylovSVD bSVD();
 *
 * arma::mat u, s, v;
 *
 * // Use the Apply() method to get a factorization.
 * bSVD.Apply(data, u, s, v, rank);
 * @endcode
 */
class RandomizedBlockKrylovSVD
{
 public:
  /**
   * Create object for the randomized block krylov SVD method.
   *
   * @param data Data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param s Diagonal matrix of singular values.
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   * @param rank Rank of the approximation (Default: number of rows.)
   * @param blockSize The block size, must be >= rank (Default: rank + 10).
   */
  template<typename InMatType, typename MatType, typename VecType>
  RandomizedBlockKrylovSVD(const InMatType& data,
                           MatType& u,
                           VecType& s,
                           MatType& v,
                           const size_t maxIterations = 2,
                           const size_t rank = 0,
                           const size_t blockSize = 0);

  /**
   * Create object for the randomized block krylov SVD method.
   *
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   * @param blockSize The block size, must be >= rank (Default: rank + 10).
   */
  RandomizedBlockKrylovSVD(const size_t maxIterations = 2,
                           const size_t blockSize = 0);

  /**
   * Apply Principal Component Analysis to the provided data set using the
   * randomized block krylov SVD.
   *
   * @param data Data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param s Diagonal matrix of singular values.
   * @param rank Rank of the approximation.
   */
  template<typename InMatType, typename MatType, typename VecType>
  void Apply(const InMatType& data,
             MatType& u,
             VecType& s,
             MatType& v,
             const size_t rank);

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

  //! The block size value.
  size_t blockSize;
};

} // namespace mlpack

// Include implementation.
#include "randomized_block_krylov_svd_impl.hpp"

#endif
