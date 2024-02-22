/**
 * @file core/dists/regression_distribution.hpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression
 * (HMMR).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTS_REGRESSION_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTS_REGRESSION_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/gaussian_distribution.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

namespace mlpack {

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
  LinearRegression<> rf;
  //! Error distribution.
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
  mlpack_deprecated RegressionDistribution(const arma::mat& predictors,
                                           const arma::vec& responses) :
    RegressionDistribution(predictors, arma::rowvec(responses.t()))
  {}

  /**
   * Create a Conditional Gaussian distribution with conditional mean function
   * obtained by running RegressionFunction on predictors, responses.
   *
   * @param predictors Matrix of predictors (X).
   * @param responses Vector of responses (y).
   */
  RegressionDistribution(const arma::mat& predictors,
                         const arma::rowvec& responses)
  {
    rf.Train(predictors, responses);
    err = GaussianDistribution(1);
    arma::mat cov(1, 1);
    cov(0, 0) = rf.ComputeError(predictors, responses);
    err.Covariance(std::move(cov));
  }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rf));
    ar(CEREAL_NVP(err));
  }

  //! Return regression function.
  const LinearRegression<>& Rf() const { return rf; }
  //! Modify regression function.
  LinearRegression<>& Rf() { return rf; }

  //! Return error distribution.
  const GaussianDistribution& Err() const { return err; }
  //! Modify error distribution.
  GaussianDistribution& Err() { return err; }

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Train(const arma::mat& observations);

  /**
   * Estimate parameters using provided observation weights.
   *
   * @param observations List of observations.
   * @param weights Probability that given observation is from distribution.
   */
  mlpack_deprecated void Train(const arma::mat& observations,
                               const arma::vec& weights);

  /**
   * Estimate parameters using provided observation weights.
   *
   * @param observations List of observations.
   * @param weights Probability that given observation is from distribution.
   */
  void Train(const arma::mat& observations, const arma::rowvec& weights);

  /**
   * Evaluate probability density function of given observation.
   *
   * @param observation Point to evaluate probability at.
   */
  double Probability(const arma::vec& observation) const;

  /**
   * Evaluate log probability density function of given observation.
   *
   * @param observation Point to evaluate log probability at.
   */
  double LogProbability(const arma::vec& observation) const
  {
    return std::log(Probability(observation));
  }

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points The data points to calculate with.
   * @param predictions Y, will contain calculated values on completion.
   */
  mlpack_deprecated void Predict(const arma::mat& points,
                                 arma::vec& predictions) const;

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points The data points to calculate with.
   * @param predictions Y, will contain calculated values on completion.
   */
  void Predict(const arma::mat& points, arma::rowvec& predictions) const;

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return rf.Parameters(); }

  //! Return the dimensionality.
  size_t Dimensionality() const { return rf.Parameters().n_elem; }
};


} // namespace mlpack

// Include implementation.
#include "regression_distribution_impl.hpp"

#endif
