/**
 * @file methods/block_krylov_svd/randomized_block_krylov_svd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the randomized block krylov SVD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BLOCK_KRYLOV_SVD_RANDOMIZED_BLOCK_KRYLOV_SVD_IMPL_HPP
#define MLPACK_METHODS_BLOCK_KRYLOV_SVD_RANDOMIZED_BLOCK_KRYLOV_SVD_IMPL_HPP

#include "randomized_block_krylov_svd.hpp"

namespace mlpack {

template<typename InMatType, typename MatType, typename VecType>
inline RandomizedBlockKrylovSVD::RandomizedBlockKrylovSVD(
    const InMatType& data,
    MatType& u,
    VecType& s,
    MatType& v,
    const size_t maxIterations,
    const size_t rank,
    const size_t blockSize) :
    maxIterations(maxIterations),
    blockSize(blockSize)
{
  if (rank == 0)
  {
    Apply(data, u, s, v, data.n_rows);
  }
  else
  {
    Apply(data, u, s, v, rank);
  }
}

inline RandomizedBlockKrylovSVD::RandomizedBlockKrylovSVD(
    const size_t maxIterations,
    const size_t blockSize) :
    maxIterations(maxIterations),
    blockSize(blockSize)
{
  /* Nothing to do here */
}

template<typename InMatType, typename MatType, typename VecType>
inline void RandomizedBlockKrylovSVD::Apply(const InMatType& data,
                                            MatType& u,
                                            VecType& s,
                                            MatType& v,
                                            const size_t rank)
{
  MatType Q, R, block, blockIteration;

  if (blockSize == 0)
  {
    // The block size cannot be greater than the number of points in the
    // dataset or the dimensionality of the dataset.
    blockSize = std::min((size_t) data.n_rows, std::min((size_t) data.n_cols,
        rank + 10));
  }

  // Random block initialization.
  MatType G = arma::randn<MatType>(data.n_cols, blockSize);

  // Construct and orthonormalize Krylov subspace.
  MatType K(data.n_rows, blockSize * (maxIterations + 1));

  // Create a working matrix using data from writable auxiliary memory
  // (K matrix). Doing so avoids an unnecessary copy in upcoming step.
  MakeAlias(block, K, data.n_rows, blockSize, false);
  arma::qr_econ(block, R, data * G);

  for (size_t blockOffset = block.n_elem; blockOffset < K.n_elem;
      blockOffset += block.n_elem)
  {
    // Temporary working matrix to store the result in the correct place.
    MakeAlias(blockIteration, K, block.n_rows, block.n_cols, blockOffset,
        false);

    arma::qr_econ(blockIteration, R, data * (data.t() * block));

    // Update working matrix for the next iteration.
    MakeAlias(block, K, block.n_rows, block.n_cols, blockOffset,
        false);
  }

  arma::qr_econ(Q, R, K);

  // Approximate eigenvalues and eigenvectors using Rayleigh-Ritz method.
  arma::svd_econ(u, s, v, Q.t() * data);

  // Do economical singular value decomposition and compute only the
  // approximations of the left singular vectors by using the centered data
  // applied to Q.
  u = Q * u;
}

} // namespace mlpack

#endif
