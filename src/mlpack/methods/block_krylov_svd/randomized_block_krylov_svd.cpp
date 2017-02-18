/**
 * @file randomized_block_krylov_svd.cpp
 * @author Marcus Edel
 *
 * Implementation of the randomized SVD method.
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
  arma::mat Q, R, block;

  if (blockSize == 0)
  {
    blockSize = rank + 10;
  }

  // Random block initialization.
  arma::mat G = arma::randn(data.n_rows, blockSize);

  // Construct and orthonormalize Krlov subspace.
  arma::mat K(data.n_rows, blockSize * (maxIterations + 1));
  arma::qr_econ(block, R, data * G);

  // Copy the temporary memory to the right place.
  K.submat(0, 0, block.n_rows - 1, block.n_cols - 1) = block;

  for (size_t i = 0, b = block.n_cols; i < maxIterations; ++i,
      b += block.n_cols)
  {
    arma::qr_econ(block, R, data * (data.t() * block));
    K.submat(0, b, block.n_rows - 1, b + block.n_cols - 1) = block;
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
