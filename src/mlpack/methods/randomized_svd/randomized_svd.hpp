/**
 * @file methods/randomized_svd/randomized_svd.hpp
 * @author Marcus Edel
 *
 * An implementation of the randomized SVD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RANDOMIZED_SVD_RANDOMIZED_SVD_HPP
#define MLPACK_METHODS_RANDOMIZED_SVD_RANDOMIZED_SVD_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * Randomized SVD is a matrix factorization that is based on randomized matrix
 * approximation techniques, developed in in "Finding structure with randomness:
 * Probabilistic algorithms for constructing approximate matrix decompositions".
 *
 * For more information, see the following.
 *
 * @code
 * @article{Halko2011,
 *   author  = {Halko, N. and Martinsson, P. G. and Tropp, J. A.},
 *   title   = {Finding Structure with Randomness: Probabilistic Algorithms for
                Constructing Approximate Matrix Decompositions},
 *   journal = {SIAM Rev.},
 *   volume  = {53},
 *   year    = {2011},
 * }
 * @endcode
 *
 * @code
 * @article{Szlam2014,
 *   author  = {Arthur Szlam Yuval Kluger and Mark Tygert},
 *   title   = {An implementation of a randomized algorithm for principal
                component analysis},
 *   journal = {CoRR},
 *   volume  = {abs/1412.3510},
 *   year    = {2014},
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
 * // Make a RandomizedSVD object.
 * RandomizedSVD rSVD();
 *
 * arma::mat u, s, v;
 *
 * // Use the Apply() method to get a factorization.
 * rSVD.Apply(data, u, s, v, rank);
 * @endcode
 */
class RandomizedSVD
{
 public:
  /**
   * Create object for the randomized SVD method.
   *
   * @param data Data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param s Diagonal "Sigma" matrix of singular values.
   * @param iteratedPower Size of the normalized power iterations
   *        (Default: rank + 2).
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   * @param rank Rank of the approximation (Default: number of rows.)
   * @param eps The eps coefficient to avoid division by zero (numerical
   *        stability).
   */
  RandomizedSVD(const arma::mat& data,
                arma::mat& u,
                arma::vec& s,
                arma::mat& v,
                const size_t iteratedPower = 0,
                const size_t maxIterations = 2,
                const size_t rank = 0,
                const double eps = 1e-7);

  /**
   * Create object for the randomized SVD method.
   *
   * @param iteratedPower Size of the normalized power iterations
   *        (Default: rank + 2).
   * @param maxIterations Number of iterations for the power method
   *        (Default: 2).
   * @param eps The eps coefficient to avoid division by zero (numerical
   *        stability).
   */
  RandomizedSVD(const size_t iteratedPower = 0,
                const size_t maxIterations = 2,
                const double eps = 1e-7);

/**
   * Center the data to apply Principal Component Analysis on given sparse
   * matrix dataset using randomized SVD.
   *
   * @param data Sparse data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param s Diagonal "Sigma" matrix of singular values.
   * @param rank Rank of the approximation.
   */
  void Apply(const arma::sp_mat& data,
             arma::mat& u,
             arma::vec& s,
             arma::mat& v,
             const size_t rank);

/**
   * Center the data to apply Principal Component Analysis on given matrix
   * dataset using randomized SVD.
   *
   * @param data Data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param s Diagonal "Sigma" matrix of singular values.
   * @param rank Rank of the approximation.
   */
  void Apply(const arma::mat& data,
             arma::mat& u,
             arma::vec& s,
             arma::mat& v,
             const size_t rank);

  /**
   * Apply Principal Component Analysis to the provided matrix data set
   * using the randomized SVD.
   *
   * @param data Data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param s Diagonal "Sigma" matrix of singular values.
   * @param rank Rank of the approximation.
   * @param rowMean Centered mean value matrix.
   */
  template<typename MatType>
  void Apply(const MatType& data,
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

  //! Get the size of the normalized power iterations.
  size_t IteratedPower() const { return iteratedPower; }
  //! Modify the size of the normalized power iterations.
  size_t& IteratedPower() { return iteratedPower; }

  //! Get the number of iterations for the power method.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the number of iterations for the power method.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the value used for decomposition stability.
  double Epsilon() const { return eps; }
  //! Modify the value used for decomposition stability.
  double& Epsilon() { return eps; }

 private:
  //! Locally stored size of the normalized power iterations.
  size_t iteratedPower;

  //! Locally stored number of iterations for the power method.
  size_t maxIterations;

  //! The value used for numerical stability.
  double eps;
};

} // namespace mlpack

// Include implementation.
#include "randomized_svd_impl.hpp"

#endif
