/**
 * @file gaussian_init.hpp
 * @author Kris Singh
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a gaussian matrix with a given mean and variance
 * to the weight matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_GAUSSIAN_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_GAUSSIAN_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize weigth matrix with a gaussian.
 */
class GaussianInitialization
{
 public:
  /**
   * Initialize the gaussian initialization rule with mean=0 and
   * variance = 1.
   *
   * @param mean The mean of the gaussian.
   * @param variance The variance of the gaussian.
   */
  GaussianInitialization():
  mean(arma::zeros<arma::vec>(1)), covariance(arma::eye<arma::mat>(1, 1))
  {}

  /**
   * Initialize the random initialization rule with the given bound.
   * Using the negative of the bound as lower bound and the positive bound as
   * upper bound.
   *
   * @param bound The number used as lower bound
   */
  GaussianInitialization(arma::mat& W, const size_t rows, const size_t cols):
  mean(arma::zeros<arma::vec>(rows)), covariance(arma::eye<arma::mat>(rows, rows))
  {}

  /**
   * Initialize the elements weight matrix using a Gaussian Distribution
   * of given mean and variance.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  void Initialize(arma::mat& W,
                  const size_t rows,
                  const size_t cols)
  {
    W = arma::mat(rows, cols);
    distribution::GaussianDistribution dist(mean, covariance);
    if (mean.n_elem == 1 && covariance.n_elem == 1)
    {
      W = arma::randn(size(W));
    }
    else
    {
      for (size_t i = 0; i < W.n_rows; i++)
        W.col(i) = dist.Random();
    }
  }

  /**
   * Initialize randomly the elements of the specified weight 3rd order tensor.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slice Numbers of slices.
   */
  void Initialize(arma::cube & W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::cube(rows, cols, slices);
    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  //! Mean of the gaussian.
  const arma::vec mean;

  //! Variance of the gaussian.
  const arma::mat covariance;
}; // class GaussianInitialization
} // namespace ann
} // namespace mlpack

#endif
