/**
 * @file gaussian_init.hpp
 * @author PlantsAndBuildings
 * @author Shashank Shekhar
 *
 * Intialization rule for the neural networks. This initialization is performed
 * by assigning values from a gaussian distribution with given mean and 
 * variance to the weight matrix and also making sure that the values are not
 * more than two standard deviations away from the mean.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_TRUNCATED_GAUSSIAN_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_TRUNCATED_GAUSSIAN_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

using namespace mlpack::math;

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class initializes a given weight matrix with values sampled from a
 * Gaussian Distribution of given mean and variance. It ensures that none of
 * the sampled values are more than two standard deviations away from the mean.
 */
class TruncatedGaussianInitialization
{
 public:
  /**
   * Initialize the gaussian with the given mean and variance and also store 
   * the standard deviation.
   *
   * @param mean Mean of the gaussian.
   * @param variance Variance of the gaussian.
   */
  TruncatedGaussianInitialization(const double mean = 0,
                                  const double variance = 1) :
      mean(mean), variance(variance), stddev(sqrt(variance))
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements weight matrix using a Truncated Gaussian 
   * Distribution
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  void Initialize(arma::mat& W,
                  const size_t rows,
                  const size_t cols)
  {
    if (W.is_empty())
    {
      W = arma::mat(rows, cols);
    }
    W.imbue(
        [&]()
        {
          double candidate = 3.0*stddev;
          while (std::abs(candidate - mean) > 2.0*stddev)
            candidate = arma::as_scalar(RandNormal(mean, variance));
          return candidate;
        });
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
  double mean;

  //! Variance of the gaussian.
  double variance;

  //! Standard deviation of the gaussian.
  double stddev;
}; // class TruncatedGaussianInitialization

} // namespace ann
} // namespace mlpack

#endif
