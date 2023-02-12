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

template<typename MatType>
inline void RandomizedSVD::Apply(const MatType& data,
                                 arma::mat& u,
                                 arma::vec& s,
                                 arma::mat& v,
                                 const size_t rank,
                                 MatType rowMean)
{
  if (iteratedPower == 0)
      iteratedPower = rank + 2;

  arma::mat R, Q, Qdata;

  // Apply the centered data matrix to a random matrix, obtaining Q.
  if (data.n_cols >= data.n_rows)
  {
    R = arma::randn<arma::mat>(data.n_rows, iteratedPower);
    Q = (data.t() * R) - arma::repmat(arma::trans(R.t() * rowMean),
        data.n_cols, 1);
  }
  else
  {
    R = arma::randn<arma::mat>(data.n_cols, iteratedPower);
    Q = (data * R) - (rowMean * (arma::ones(1, data.n_cols) * R));
  }

  // Form a matrix Q whose columns constitute a
  // well-conditioned basis for the columns of the earlier Q.
  if (maxIterations == 0)
  {
  arma::qr_econ(Q, v, Q);
  }
  else
  {
    arma::lu(Q, v, Q);
  }

  // Perform normalized power iterations.
  for (size_t i = 0; i < maxIterations; ++i)
  {
    if (data.n_cols >= data.n_rows)
    {
      Q = (data * Q) - rowMean * (arma::ones(1, data.n_cols) * Q);
      arma::lu(Q, v, Q);
      Q = (data.t() * Q) - arma::repmat(rowMean.t() * Q, data.n_cols, 1);
    }
    else
    {
      Q = (data.t() * Q) - arma::repmat(rowMean.t() * Q, data.n_cols, 1);
      arma::lu(Q, v, Q);
      Q = (data * Q) - (rowMean * (arma::ones(1, data.n_cols) * Q));
    }

    // Computing the LU decomposition is more efficient than computing the QR
    // decomposition, so we only use it in the last iteration, a pivoted QR
    // decomposition which renormalizes Q, ensuring that the columns of Q are
    // orthonormal.
    if (i < (maxIterations - 1))
    {
      arma::lu(Q, v, Q);
    }
    else
    {
      arma::qr_econ(Q, v, Q);
    }
  }

  // Do economical singular value decomposition and compute only the
  // approximations of the left singular vectors by using the centered data
  // applied to Q.
  if (data.n_cols >= data.n_rows)
  {
    Qdata = (data * Q) - rowMean * (arma::ones(1, data.n_cols) * Q);
    arma::svd_econ(u, s, v, Qdata);
    v = Q * v;
  }
  else
  {
    Qdata = (Q.t() * data) - arma::repmat(Q.t() * rowMean, 1,  data.n_cols);
    arma::svd_econ(u, s, v, Qdata);
    u = Q * u;
  }
}

} // namespace mlpack

#endif
