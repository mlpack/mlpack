/**
 * @file core/distributions/discrete_distribution.hpp
 * @author Ryan Curtin
 * @author Rohan Raj
 *
 * Implementation of the discrete distribution, where each discrete observation
 * has a given probability.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {

/**
 * A discrete distribution where the only observations are discrete
 * observations.  This is useful (for example) with discrete Hidden Markov
 * Models, where observations are non-negative integers representing specific
 * emissions.
 *
 * No bounds checking is performed for observations, so if an invalid
 * observation is passed (i.e. observation > numObservations), a crash will
 * probably occur.
 *
 * @note
 * This class, like every other class in mlpack, uses arma::vec to represent
 * observations.  While a discrete distribution only has positive integers
 * (size_t) as observations, these can be converted to doubles (which is what
 * arma::vec holds).  This distribution internally converts those doubles back
 * into size_t before comparisons.
 */
class DiscreteDistribution
{
 public:
  /**
   * Default constructor, which creates a distribution that has no
   * observations.
   */
  DiscreteDistribution() :
      probabilities(std::vector<arma::vec>(1)){ /* Nothing to do. */ }

  /**
   * Define the discrete distribution as having numObservations possible
   * observations.  The probability in each state will be set to (1 /
   * numObservations).
   *
   * @param numObservations Number of possible observations this distribution
   *    can have.
   */
  DiscreteDistribution(const size_t numObservations) :
      probabilities(std::vector<arma::vec>(1,
          arma::ones<arma::vec>(numObservations) / numObservations))
  { /* Nothing to do. */ }

  /**
   * Define the multidimensional discrete distribution as having
   * numObservations possible observations.  The probability in each state will
   * be set to (1 / numObservations of each dimension).
   *
   * @param numObservations Number of possible observations this distribution
   *    can have.
   */
  DiscreteDistribution(const arma::Col<size_t>& numObservations)
  {
    for (size_t i = 0; i < numObservations.n_elem; ++i)
    {
      const size_t numObs = size_t(numObservations[i]);
      if (numObs <= 0)
      {
        std::ostringstream oss;
        oss << "number of observations for dimension " << i << " is 0, but "
            << "must be greater than 0";
        throw std::invalid_argument(oss.str());
      }
      probabilities.push_back(arma::ones<arma::vec>(numObs) / numObs);
    }
  }

  /**
   * Define the multidimensional discrete distribution as having the given
   * probabilities for each observation.
   *
   * @param probabilities Probabilities of each possible observation.
   */
  DiscreteDistribution(const std::vector<arma::vec>& probabilities)
  {
    for (size_t i = 0; i < probabilities.size(); ++i)
    {
      arma::vec temp = probabilities[i];
      double sum = accu(temp);
      if (sum > 0)
        this->probabilities.push_back(temp / sum);
      else
      {
        this->probabilities.push_back(arma::ones<arma::vec>(temp.n_elem)
            / temp.n_elem);
      }
    }
  }

  /**
   * Get the dimensionality of the distribution.
   */
  size_t Dimensionality() const { return probabilities.size(); }

  /**
   * Return the probability of the given observation.  If the observation is
   * greater than the number of possible observations, then a crash will
   * probably occur -- bounds checking is not performed.
   *
   * @param observation Observation to return the probability of.
   * @return Probability of the given observation.
   */
  double Probability(const arma::vec& observation) const
  {
    double probability = 1.0;
    // Ensure the observation has the same dimension with the probabilities.
    if (observation.n_elem != probabilities.size())
    {
      Log::Fatal << "DiscreteDistribution::Probability(): observation has "
          << "incorrect dimension " << observation.n_elem << " but should have"
          << " dimension " << probabilities.size() << "!" << std::endl;
    }

    for (size_t dimension = 0; dimension < observation.n_elem; dimension++)
    {
      // Adding 0.5 helps ensure that we cast the floating point to a size_t
      // correctly.
      const size_t obs = size_t(observation(dimension) + 0.5);

      // Ensure that the observation is within the bounds.
      if (obs >= probabilities[dimension].n_elem)
      {
        Log::Fatal << "DiscreteDistribution::Probability(): received "
            << "observation " << obs << "; observation must be in [0, "
            << probabilities[dimension].n_elem << "] for this distribution."
            << std::endl;
      }
      probability *= probabilities[dimension][obs];
    }

    return probability;
  }

  /**
   * Return the log probability of the given observation.  If the observation
   * is greater than the number of possible observations, then a crash will
   * probably occur -- bounds checking is not performed.
   *
   * @param observation Observation to return the log probability of.
   * @return Log probability of the given observation.
   */
  double LogProbability(const arma::vec& observation) const
  {
    // TODO: consider storing log probabilities instead?
    return std::log(Probability(observation));
  }

  /**
   * Calculates the Discrete probability density function for each
   * data point (column) in the given matrix.
   *
   * @param x List of observations.
   * @param probabilities Output probabilities for each input observation.
   */
  void Probability(const arma::mat& x, arma::vec& probabilities) const
  {
    probabilities.set_size(x.n_cols);
    for (size_t i = 0; i < x.n_cols; ++i)
      probabilities(i) = Probability(x.unsafe_col(i));
  }

  /**
   * Returns the Log probability of the given matrix. These values are stored
   * in logProbabilities.
   *
   * @param x List of observations.
   * @param logProbabilities Output log-probabilities for each input
   *   observation.
   */
  void LogProbability(const arma::mat& x, arma::vec& logProbabilities) const
  {
    logProbabilities.set_size(x.n_cols);
    for (size_t i = 0; i < x.n_cols; ++i)
      logProbabilities(i) = std::log(Probability(x.unsafe_col(i)));
  }

  /**
   * Return a randomly generated observation (one-dimensional vector; one
   * observation) according to the probability distribution defined by this
   * object.
   *
   * @return Random observation.
   */
  arma::vec Random() const;

  /**
   * Estimate the probability distribution directly from the given
   * observations. If any of the observations is greater than numObservations,
   * a crash is likely to occur.
   *
   * @param observations List of observations.
   */
  void Train(const arma::mat& observations);

  /**
   * Estimate the probability distribution from the given observations, taking
   * into account the probability of each observation actually being from this
   * distribution.
   *
   * @param observations List of observations.
   * @param probabilities List of probabilities that each observation is
   *     actually from this distribution.
   */
  void Train(const arma::mat& observations,
             const arma::vec& probabilities);

  //! Return the vector of probabilities for the given dimension.
  arma::vec& Probabilities(const size_t dim = 0) { return probabilities[dim]; }
  //! Modify the vector of probabilities for the given dimension.
  const arma::vec& Probabilities(const size_t dim = 0) const
  { return probabilities[dim]; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(probabilities));
  }

 private:
  //! The probabilities for each dimension; each arma::vec represents the
  //! probabilities for the observations in each dimension.
  std::vector<arma::vec> probabilities;
};

} // namespace mlpack

// Include implementation.
#include "discrete_distribution_impl.hpp"

#endif
