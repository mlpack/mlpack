/**
 * @file regression_distribution.cpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression (HMMR)
 */

#include "regression_distribution.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

/**
 * Returns a string representation of this object.
 */
std::string RegressionDistribution::ToString() const
{
  std::ostringstream convert;
  convert << "HMMRegression [" << this << "]" << std::endl;

  // Secondary ostringstream so things can be indented right.
  std::ostringstream data;
  data << "Conditional mean function: " << std::endl << rf.ToString();
  data << "Parameters: " << std::endl << rf.Parameters();
  data << "Error distribution: " << std::endl << err.ToString();

  convert << util::Indent(data.str());
  return convert.str();
}

/**
 * Estimate parameters using provided observation weights
 *
 * @param observations List of observations.
 */
void RegressionDistribution::Estimate(const arma::mat& observations)
{
  regression::LinearRegression lr(observations.rows(1, observations.n_rows - 1),
      (observations.row(0)).t(), 0, true);
  rf = lr;
  arma::vec fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Estimate(observations.row(0) - fitted.t());
}

/**
 * Estimate parameters using provided observation weights.
 *
 * @param weights probability that given observation is from distribution
 */
void RegressionDistribution::Estimate(const arma::mat& observations,
                             const arma::vec& weights)
{
  regression::LinearRegression lr(observations.rows(1, observations.n_rows - 1),
      (observations.row(0)).t(), 0, true, weights);
  rf = lr;
  arma::vec fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Estimate(observations.row(0) - fitted.t(), weights);
}

/**
 * Evaluate probability density function of given observation.
 *
 * @param observation point to evaluate probability at
 */
double RegressionDistribution::Probability(const arma::vec& observation) const
{
  arma::vec fitted;
  rf.Predict(observation.rows(1, observation.n_rows-1), fitted);
  return err.Probability(observation(0)-fitted);
}

void RegressionDistribution::Predict(const arma::mat& points,
                                     arma::vec& predictions) const
{
  rf.Predict(points, predictions);
}
