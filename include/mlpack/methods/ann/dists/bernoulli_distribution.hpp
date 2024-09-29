/**
 * @file methods/ann/dists/bernoulli_distribution.hpp
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

/**
 * Multiple independent Bernoulli distributions.
 *
 * Bernoulli distribution is the discrete probability distribution of a random
 * variable which takes the value 1 with probability p and the value 0 with
 * probability q = 1 - p.
 * In this implementation, the p values of the distributions are given by the
 * param matrix.
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
   * Create multiple independent Bernoulli distributions whose p values are
   * given by the param parameter. Thus, we create nofRows * nofColumns
   * distributions. The shape of the matrix of distributions is the same as the
   * shape of the param matrix as each element of the param matrix parameterizes
   * one Bernoulli distribution.
   * This is used in the ANN module to define distribution for each feature in
   * each batch, where number of features becomes nofRows and batch size becomes
   * nofColumns.
   *
   * applyLogistic has to be true if all the elements of param matrix are not
   * in the range [0, 1].
   *
   * @param param The matrix of probabilities or pre probabilities of
   *        the multiple distributions.
   * @param applyLogistic If true, we apply Logistic function to the param
   *        matrix (pre probability) to get probability.
   * @param eps The minimum value used for computing logarithms and
   *        denominators.
   */
  BernoulliDistribution(const DataType& param,
                        const bool applyLogistic = true,
                        const double eps = 1e-10);

  /**
   * Return the probabilities of the given matrix of observations.
   *
   * @param observation The observation matrix.
   */
  double Probability(const DataType& observation) const
  {
    return std::exp(LogProbability(observation));
  }

  /**
   * Return the log probabilities of the given matrix of observations.
   *
   * @param observation The observation matrix.
   */
  double LogProbability(const DataType& observation) const;

  /**
   * Stores the gradient of the log probabilities of the observations in the
   * output matrix.
   *
   * @param observation The observation matrix.
   * @param output The output matrix where the gradients are stored.
   */
  void LogProbBackward(const DataType& observation, DataType& output) const;

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

  //! Return the logits matrix.
  const DataType& Logits() const { return logits; }

  //! Return a modifiable copy of the pre probability matrix.
  DataType& Logits() { return logits; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    // We just need to serialize each of the members.
    ar(CEREAL_NVP(probability));
    ar(CEREAL_NVP(logits));
    ar(CEREAL_NVP(applyLogistic));
    ar(CEREAL_NVP(eps));
  }

 private:
  //! Probabilities of the distributions.
  DataType probability;

  //! logits matrix of the distributions. After applying logistic function, it
  //! gives probability matrix.
  DataType logits;

  //! If true, apply logistic function to probability matrix.
  bool applyLogistic;

  //! The minimum value used for computing logarithms and denominators.
  double eps;
}; // class BernoulliDistribution

} // namespace mlpack

// Include implementation.
#include "bernoulli_distribution_impl.hpp"

#endif
