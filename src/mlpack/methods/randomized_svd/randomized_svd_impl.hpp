/**
 * @file methods/randomized_svd/randomized_svd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the randomized SVD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RANDOMIZED_SVD_RANDOMIZED_SVD_IMPL_HPP
#define MLPACK_METHODS_RANDOMIZED_SVD_RANDOMIZED_SVD_IMPL_HPP

#include "randomized_svd.hpp"

namespace mlpack {

inline RandomizedSVD::RandomizedSVD(
    const arma::mat& data,
    arma::mat& u,
    arma::vec& s,
    arma::mat& v,
    const size_t iteratedPower,
    const size_t maxIterations,
    const size_t rank,
    const double eps) :
    iteratedPower(iteratedPower),
    maxIterations(maxIterations),
    eps(eps)
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

inline RandomizedSVD::RandomizedSVD(
    const size_t iteratedPower,
    const size_t maxIterations,
    const double eps) :
    iteratedPower(iteratedPower),
    maxIterations(maxIterations),
    eps(eps)
{
  /* Nothing to do here */
}


inline void RandomizedSVD::Apply(const arma::sp_mat& data,
                                 arma::mat& u,
                                 arma::vec& s,
                                 arma::mat& v,
                                 const size_t rank)
{
  // Center the data into a temporary matrix for sparse matrix.
  arma::sp_mat rowMean = arma::sum(data, 1) / data.n_cols;

  Apply(data, u, s, v, rank, rowMean);
}

inline void RandomizedSVD::Apply(const arma::mat& data,
                                 arma::mat& u,
                                 arma::vec& s,
                                 arma::mat& v,
                                 const size_t rank)
{
  // Center the data into a temporary matrix.
  arma::mat rowMean = arma::sum(data, 1) / data.n_cols + eps;

  Apply(data, u, s, v, rank, rowMean);
}

} // namespace mlpack

#endif
