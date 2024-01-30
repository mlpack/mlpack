/**
 * @file methods/ann/dists/normal_distribution.hpp
 * @author xiaohong ji
 * @author Nishant Kumar
 *
 * Definition of the Normal distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_HPP
#define MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>
#include "../activation_functions/logistic_function.hpp"

namespace mlpack {

/**
 * Implementation of the Normal Distribution function.
 *
 * Normal distribution is a function which accepts a mean and a standard deviation
 * term and creates a probability distribution out of it.
 * 
 * @tparam DataType Type of the input data. (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <typename DataType = arma::mat>
class NormalDistribution
{
 public:
  /**
   * Default constructor, which creates a Normal distribution with zero
   * dimension.
   */
  NormalDistribution();

  /**
   * Create a Normal distribution with the given mean and sigma.
   *
   * @param mean The mean of the normal distribution.
   * @param sigma The standard deviation of the normal distribution.
   */
  NormalDistribution(const DataType& mean, const DataType& sigma);

  /**
   * Return the probabilities of the given matrix of observations.
   *
   * @param observation The observation matrix.
   */
  DataType Probability(const DataType& observation) const
  {
    return exp(LogProbability(observation));
  }

  /**
   * Return the log probabilities of the given matrix of observations.
   *
   * @param observation The observation matrix.
   */
  DataType LogProbability(const DataType& observation) const;

  /**
   * Stores the gradient of the probabilities of the observations
   * with respect to mean and standard deviation.
   *
   * @param observation The observation matrix.
   * @param dmu The gradient with respect to mean.
   * @param dsigma The gradient with respect to the standard deviation.
   */
  void ProbBackward(const DataType& observation,
                    DataType& dmu,
                    DataType& dsigma) const;

  /**
   * Calculates the normal probability density function for each
   * data point (column) in the given matrix.
   *
   * @param x The observation matrix.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const DataType& x, DataType& probabilities) const
  {
    probabilities = Probability(x);
  }

  /**
    * Calculates the log of normal probability density function for each
    * data point (column) in the given matrix.
    *
    * @param x The observation matrix.
    * @param probabilities Output log probabilities for each input observation.
    */
  void LogProbability(const DataType& x, DataType& probabilities) const
  {
    probabilities = LogProbability(x);
  }

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation from this Normal distribution.
   */
  DataType Sample() const;

  //! Get the mean.
  const DataType& Mean() const { return mean; }

  //! Modify the mean.
  DataType& Mean() { return mean; }

  //! Get the standard deviation.
  const DataType& StandardDeviation() const { return sigma; }

  //! Modify the standard deviation.
  DataType& StandardDeviation() { return sigma; }

  //! Return the dimensionality of this distribution.
  size_t Dimensionality() const { return mean.n_elem; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Mean of the distribution.
  DataType mean;

  //! Standard deviation of the distribution.
  DataType sigma;
}; // class NormalDistribution

} // namespace mlpack

// Include implementation.
#include "normal_distribution_impl.hpp"

#endif
