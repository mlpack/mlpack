/**
 * @file methods/block_krylov_svd/randomized_block_krylov_svd.cpp
 * @author Marcus Edel
 *
 * Implementation of the randomized block krylov SVD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "randomized_block_krylov_svd.hpp"

namespace mlpack {
namespace svd {

RandomizedBlockKrylovSVD::RandomizedBlockKrylovSVD(const arma::mat& data,
                                                   arma::mat& u,
                                                   arma::vec& s,
                                                   arma::mat& v,
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

RandomizedBlockKrylovSVD::RandomizedBlockKrylovSVD(const size_t maxIterations,
                                                   const size_t blockSize) :
    maxIterations(maxIterations),
    blockSize(blockSize)
{
  /* Nothing to do here */
}

void RandomizedBlockKrylovSVD::Apply(const arma::mat& data,
                                     arma::mat& u,
                                     arma::vec& s,
                                     arma::mat& v,
                                     const size_t rank)
{
  arma::mat Q, R, block, blockIteration;

  if (blockSize == 0)
  {
    blockSize = rank + 10;
  }

  // Random block initialization.
  arma::mat G = arma::randn(data.n_cols, blockSize);

  // Construct and orthonormalize Krylov subspace.
  arma::mat K(data.n_rows, blockSize * (maxIterations + 1));

  // Create a working matrix using data from writable auxiliary memory
  // (K matrix). Doing so avoids an uncessary copy in upcoming step.
  block = arma::mat(K.memptr(), data.n_rows, blockSize, false, false);
  arma::qr_econ(block, R, data * G);

  for (size_t blockOffset = block.n_elem; blockOffset < K.n_elem;
      blockOffset += block.n_elem)
  {
    // Temporary working matrix to store the result in the correct place.
    blockIteration = arma::mat(K.memptr() + blockOffset, block.n_rows,
        block.n_cols, false, false);

    arma::qr_econ(blockIteration, R, data * (data.t() * block));

    // Update working matrix for the next iteration.
    block = arma::mat(K.memptr() + blockOffset, block.n_rows, block.n_cols,
        false, false);
  }

  arma::qr_econ(Q, R, K);

  // Approximate eigenvalues and eigenvectors using Rayleigh–Ritz method.
  arma::svd_econ(u, s, v, Q.t() * data);

  // Do economical singular value decomposition and compute only the
  // approximations of the left singular vectors by using the centered data
  // applied to Q.
  u = Q * u;
}

} // namespace svd
} // namespace mlpack
