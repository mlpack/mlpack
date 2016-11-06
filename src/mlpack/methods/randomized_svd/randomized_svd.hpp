/**
 * @file randomized_svd.hpp
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
namespace svd {

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
   * @param sigma Diagonal matrix of singular values.
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
   * Apply Principal Component Analysis to the provided data set using the
   * randomized SVD.
   *
   * @param data Data matrix.
   * @param u First unitary matrix.
   * @param v Second unitary matrix.
   * @param sigma Diagonal matrix of singular values.
   * @param rank Rank of the approximation.
   */
  void Apply(const arma::mat& data,
             arma::mat& u,
             arma::vec& s,
             arma::mat& v,
             const size_t rank);

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

} // namespace svd
} // namespace mlpack

#endif
