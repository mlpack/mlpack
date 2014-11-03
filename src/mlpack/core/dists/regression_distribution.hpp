/**
 * @file regression_distribution.hpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression (HMMR)
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

namespace mlpack {
namespace distribution {

/**
 * A class that represents a univariate conditionally Gaussian distribution.
 * Can be used as an emission distribution with the hmm class to implement HMM
 * regression (HMMR) as described in
 * https://www.ima.umn.edu/preprints/January1994/1195.pdf
 * The hmm observations should have the dependent variable in the first row,
 * with the independent variables in the other rows.
 */
class RegressionDistribution
{
 private:
  //! Regression function for representing conditional mean.
  regression::LinearRegression rf;
  //! Error distribution
  GaussianDistribution err;

 public:
  /**
   * Default constructor, which creates a Gaussian with zero dimension.
   */
  RegressionDistribution() { /* nothing to do */ }

  /**
   * Create a Conditional Gaussian distribution with conditional mean function
   * obtained by running RegressionFunction on predictors, responses.
   *
   * @param predictors Matrix of predictors (X).
   * @param responses Vector of responses (y).
   */
  RegressionDistribution(const arma::mat& predictors,
                         const arma::vec& responses) :
      rf(regression::LinearRegression(predictors, responses))
  {
    err = GaussianDistribution(1);
    err.Covariance() = rf.ComputeError(predictors, responses);
  }

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

  // Return regression function
  const regression::LinearRegression& Rf() {return rf;}

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Estimate(const arma::mat& observations);

  /**
   * Estimate parameters using provided observation weights
   *
   * @param weights probability that given observation is from distribution
   */
  void Estimate(const arma::mat& observations, const arma::vec& weights);

  /**
  * Evaluate probability density function of given observation
  *
  * @param observation point to evaluate probability at
  */
  double Probability(const arma::vec& observation) const;

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points the data points to calculate with.
   * @param predictions y, will contain calculated values on completion.
   */
  void Predict(const arma::mat& points, arma::vec& predictions) const;

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return rf.Parameters(); }

  //! Return the dimensionality (2) NEED TO FIX THIS
  static const size_t Dimensionality() { return 2; }
};


}; // namespace distribution
}; // namespace mlpack

#endif
