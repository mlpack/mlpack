/**
 * @file bernoulli_distribution.hpp
 * @author Atharva Khandait
 *
 * Definition of the Bernoulli distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_HPP
#define MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>
#include "../activation_functions/logistic_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Multiple Bernoulli distributions.
 *
 * @tparam DataType Type of the input data. (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <typename DataType = arma::mat>
class BernoulliDistribution
{
 public:
  /**
   * Default constructor, which creates a Bernoulli distribution with zero
   * dimension.
   */
  BernoulliDistribution();

  /**
   * Create multiple Bernoulli distributions with the given parameters.
   * If the param matrix is already between [0, 1], then applyLogistic can be
   * set to false.
   *
   * @param param The matrix of probabilities or pre probabilities of
   *        the multiple distributions.
   * @param applyLogsitic If true, then we apply Logistic function to the
   *        param matrix (pre probability) to get probability
   * @param eps The minimum value used for computing logarithms and
   *        denominators.
   */
  BernoulliDistribution(const DataType&& param,
                        const bool applyLogistic = true,
                        const double eps = 1e-10);

  /**
   * Return the probabilities of the given matrix of observations.
   *
   * @param oberservation The observation matrix.
   */
  double Probability(const DataType&& observation) const
  {
    return std::exp(LogProbability(observation));
  }

  /**
   * Return the log probabilities of the given matrix of observations.
   *
   * @param oberservation The observation matrix.
   */
  double LogProbability(const DataType&& observation) const;

  /**
   * Stores the gradient of the log probabilities of the observations in the
   * output matrix.
   *
   * @param oberservation The observation matrix.
   * @param output The output matrix where the gradients are stored.
   */
  void LogProbBackward(const DataType&& observation, DataType&& output) const;

  /**
   * Return a matrix of randomly generated samples according to the
   * probability distributions defined by this object.
   *
   * @return Matrix(integer) of random samples from the multiple Bernoulli distributions.
   */
  DataType Sample() const;

  //! Return the probability matrix.
  const DataType& Probability() const { return probability; }

  //! Return a modifiable copy of the probability matrix.
  DataType& Probability() { return probability; }

  //! Return the pre probability matrix.
  const DataType& PreProbability() const { return preProbability; }

  //! Return a modifiable copy of the pre probability matrix.
  DataType& PreProbability() { return preProbability; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    // We just need to serialize each of the members.
    ar & BOOST_SERIALIZATION_NVP(probability);
    ar & BOOST_SERIALIZATION_NVP(preProbability);
    ar & BOOST_SERIALIZATION_NVP(applyLogistic);
    ar & BOOST_SERIALIZATION_NVP(eps);
  }

 private:
  //! Probabilities of the distributions.
  DataType probability;

  //! Pre Probabilities of the distributions.
  DataType preProbability;

  //! If true, apply logistic function to probability matrix.
  bool applyLogistic;

  //! The minimum value used for computing logarithms and denominators.
  double eps;
}; // class BernoulliDistribution

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "bernoulli_distribution_impl.hpp"

#endif
