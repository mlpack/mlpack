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
  mean(0), variance(1)
  {}

  /**
   * Initialize the gaussian with the given mean and variance.
   *
   * @param mean Mean of the gaussian
   * @param variance Variance of the gaussian
   */
  GaussianInitialization(const double mean=0, const double variance=1):
  mean(mean), variance(variance)
  {}

  /**
   * Initialize the elements weight matrix using a Gaussian Distribution.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  void Initialize(arma::mat& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t seed =21)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::srand(seed);
    W = arma::mat(rows, cols);
    double m;
    double v;
    m = mean;
    v = variance;
    std::normal_distribution<> d(m, v);
    //distribution::GaussianDistribution dist(m, v);
    W.imbue( [&]() { return arma::as_scalar(d(gen)); } );
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
  const double mean;

  //! Variance of the gaussian .
  const double variance;
}; // class GaussianInitialization
} // namespace ann
} // namespace mlpack

#endif
