/**
 * @file normal_distribution.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Normal distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_NORMAL_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_NORMAL_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace distribution {

/**
 * Multiple univariate Normal(Gaussian) distributions.
 */
template <typename DataType = arma::mat>
class NormalDistribution
{
 private:
  //! Means of the distributions.
  DataType mean;
  //! Standard deviations of the distributions.
  DataType stdDeviation;

  //! log(2pi)
  static const constexpr double log2pi = 1.83787706640934533908193770912475883;

 public:
  /**
   * Default constructor, which creates a Normal distribution with zero
   * dimension.
   */
  NormalDistribution()
  {
    // Nothing to do here.
  }

  /**
   * Create multiple Normal distributions with the given means and Standard
   * deviations.
   */
  NormalDistribution(const DataType mean, const DataType stdDeviation);

  /**
   * Return the probabilities of the given matrix of observations.
   */
  double Probability(const DataType& observation) const
  {
    return exp(LogProbability(observation));
  }

  /**
   * Return the log probabilities of the given matrix of observations.
   */
  double LogProbability(const DataType& observation) const;

  /**
   *
   */
  void LogProbBackward(const DataType& observation, DataType& output) const;

  /**
   * Return a matrix of randomly generated observations according to the
   * probability distributions defined by this object.
   *
   * @return Matrix of random observations from this Normal distribution.
   */
  DataType Random() const;

  /**
   * Return the mean.
   */
  const DataType& Mean() const { return mean; }

  /**
   * Return a modifiable copy of the mean.
   */
  DataType& Mean() { return mean; }

  /**
   * Return the standard deviation.
   */
  const DataType& StdDeviation() const { return stdDeviation; }

  /**
   * Return a modifiable copy of the standard deviation.
   */
  DataType& StdDeviation() { return stdDeviation; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    // We just need to serialize each of the members.
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(stdDeviation);
  }
}; // class NormalDistribution

} // namespace distribution
} // namespace mlpack

// Include implementation.
#include "normal_distribution_impl.hpp"

#endif
