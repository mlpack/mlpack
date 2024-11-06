/**
 * @file core/distributions/regression_distribution.hpp
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
#ifndef MLPACK_CORE_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distributions/gaussian_distribution.hpp>
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
template<typename MatType = arma::mat>
class RegressionDistribution
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;
  using RowType = typename GetRowType<MatType>::type;

 private:
  //! Regression function for representing conditional mean.
  LinearRegression<MatType> rf;
  //! Error distribution.
  GaussianDistribution<MatType> err;

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
  RegressionDistribution(const MatType& predictors,
                         const RowType& responses)
  {
    rf.Train(predictors, responses);
    err = GaussianDistribution<MatType>(1);
    MatType cov(1, 1);
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
  const LinearRegression<MatType>& Rf() const { return rf; }
  //! Modify regression function.
  LinearRegression<MatType>& Rf() { return rf; }

  //! Return error distribution.
  const GaussianDistribution<MatType>& Err() const { return err; }
  //! Modify error distribution.
  GaussianDistribution<MatType>& Err() { return err; }

  /**
   * Estimate the Gaussian distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Train(const MatType& observations);

  /**
   * Estimate parameters using provided observation weights.
   *
   * @param observations List of observations.
   * @param weights Probability that given observation is from distribution.
   */
  void Train(const MatType& observations, const RowType& weights);

  /**
   * Evaluate probability density function of given observation.
   *
   * @param observation Point to evaluate probability at.
   */
  ElemType Probability(const VecType& observation) const;

  /**
   * Evaluate probability density function for the given observations.
   *
   * @param observations Points to evaluate probability at.
   * @param probabilities Vector to store computed probabilities in.
   */
  void Probability(const MatType& observations,
                   VecType& probabilities) const;

  /**
   * Evaluate log probability density function of given observation.
   *
   * @param observation Point to evaluate log probability at.
   */
  ElemType LogProbability(const VecType& observation) const
  {
    return std::log(Probability(observation));
  }

  /**
   * Evaluate log probability density function on the given observations.
   *
   * @param observations Points to evaluate log probability at.
   */
  void LogProbability(const MatType& observations,
                      VecType& probabilities) const
  {
    Probability(observations, probabilities);
    probabilities = arma::log(probabilities);
  }

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points The data points to calculate with.
   * @param predictions Y, will contain calculated values on completion.
   */
  void Predict(const MatType& points, RowType& predictions) const;

  //! Return the parameters (the b vector).
  const VecType& Parameters() const { return rf.Parameters(); }

  //! Return the dimensionality.
  size_t Dimensionality() const { return rf.Parameters().n_elem; }
};


} // namespace mlpack

// Include implementation.
#include "regression_distribution_impl.hpp"

#endif
