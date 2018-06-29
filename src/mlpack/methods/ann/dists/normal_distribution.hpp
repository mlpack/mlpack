/**
 * @file normal_distribution.hpp
 * @author Atharva Khandait
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
#include "../activation_functions/softplus_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Multiple univariate Normal(Gaussian) distributions.
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
  NormalDistribution()
  {
    // Nothing to do here.
  }

  /**
   * Create multiple Normal distributions with the given parameters.
   *
   * @param mean The DataType of means of the multiple distributions
   * @param stdDev The DataType of standard deviations of the multiple
   *        distributions.
   */
  NormalDistribution(const DataType&& mean, const DataType&& stdDev);

  /*
   * Create multiple Normal distributions with the given data.
   * Originally designed to be used along with the ReconstructionLoss class.
   * The target of the loss function which will be a single matrix can directly
   * be passed on to this function.
   *
   * @param param The DataType which has means in the lower half.
   *        The upper half has pre standard deviations.
   * @param applySoftplus If true, after applying softplus function to the pre
   *        standard deviations, we get the standard deviations.
   */
  NormalDistribution(const DataType&& param, const bool applySoftplus = true);

  /**
   * Return the probabilities of the given DataType of observations.
   */
  double Probability(const DataType&& observation) const
  {
    return exp(LogProbability(observation));
  }

  /**
   * Return the log probabilities of the given DataType of observations.
   */
  double LogProbability(const DataType&& observation) const;

  /**
   * Stores the gradient of the log probabilities of the observations in the
   * output DataType.
   */
  void LogProbBackward(const DataType&& observation, DataType&& output) const;

  /**
   * Return a matrix of randomly generated samples according to the
   * probability distributions defined by this object.
   *
   * @return Matrix of random samples from the multiple Normal distributions.
   */
  DataType Sample() const;

  //! Return the mean.
  const DataType& Mean() const { return mean; }

  //! Return a modifiable copy of the mean.
  DataType& Mean() { return mean; }

  //! Return the standard deviation.
  const DataType& StdDev() const { return stdDev; }

  //! Return a modifiable copy of the standard deviation.
  DataType& StdDev() { return stdDev; }

  //! Return the pre standard deviation.
  const DataType& PreStdDev() const { return preStdDev; }

  //! Return a modifiable copy of the pre standard deviation.
  DataType& PreStdDev() { return preStdDev; }

  //! Calculate standard deviation from the pre standard deviation.
  //! Only meant for testing.
  void ApplySoftplus() { SoftplusFunction::Fn(preStdDev, stdDev); }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    // We just need to serialize each of the members.
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(stdDev);
    ar & BOOST_SERIALIZATION_NVP(preStdDev);
  }

 private:
  //! Means of the distributions.
  DataType mean;

  //! Standard deviations of the distributions.
  DataType stdDev;

  //! Pre standard deviation. After softplus this will give standard deviation.
  DataType preStdDev;

  //! If true, apply softplus function to upper half of param matrix.
  bool applySoftplus;

  //! log(2pi)
  static const constexpr double log2pi = 1.83787706640934533908193770912475883;
}; // class NormalDistribution

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "normal_distribution_impl.hpp"

#endif
