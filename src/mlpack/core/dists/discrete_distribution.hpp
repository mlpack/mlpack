/**
 * @file discrete_distribution.hpp
 * @author Ryan Curtin
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
namespace distribution /** Probability distributions. */ {

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
 * This distribution only supports one-dimensional observations, so when passing
 * an arma::vec as an observation, it should only have one dimension
 * (vec.n_rows == 1).  Any additional dimensions will simply be ignored.
 *
 * @note
 * This class, like every other class in mlpack, uses arma::vec to represent
 * observations.  While a discrete distribution only has positive integers
 * (size_t) as observations, these can be converted to doubles (which is what
 * arma::vec holds).  This distribution internally converts those doubles back
 * into size_t before comparisons.
 * @endnote
 */
class DiscreteDistribution
{
 public:
  /**
   * Default constructor, which creates a distribution that has no observations.
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
   * Define the multidimensional discrete distribution as having numObservations possible
   * observations.  The probability in each state will be set to (1 /
   * numObservations of each dimension).
   *
   * @param numObservations Number of possible observations this distribution
   *    can have.
   */
  DiscreteDistribution(const arma::Col<size_t>& numObservations)
  {
    for (size_t i = 0; i < numObservations.n_elem; i++)
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
   * Define the multidimensional discrete distribution as having the given probabilities for each
   * observation.
   *
   * @param probabilities Probabilities of each possible observation.
   */
  DiscreteDistribution(const std::vector<arma::vec>& probabilities)
  {
    for (size_t i = 0; i < probabilities.size(); i++)
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
    // Ensure the observation has the same dimension with the probabilities
    if (observation.n_elem != probabilities.size())
    {
      Log::Debug << "the obversation must has the same dimension with the probabilities"
          << "the observation's dimension is" << observation.n_elem << "but the dimension of "
          << "probabilities is" << probabilities.size() << std::endl;
      return probability;
    }
    for (size_t dimension = 0; dimension < observation.n_elem; dimension++)
    {
      // Adding 0.5 helps ensure that we cast the floating point to a size_t
      // correctly.
      const size_t obs = size_t(observation(dimension) + 0.5);

      // Ensure that the observation is within the bounds.
      if (obs >= probabilities[dimension].n_elem)
      {
        Log::Debug << "DiscreteDistribution::Probability(): received observation "
             << obs << "; observation must be in [0, " << probabilities[dimension].n_elem
             << "] for this distribution." << std::endl;
      }
      probability *= probabilities[dimension][obs];
    }

    return probability;
  }

  /**
   * Return the log probability of the given observation.  If the observation is
   * greater than the number of possible observations, then a crash will
   * probably occur -- bounds checking is not performed.
   *
   * @param observation Observation to return the log probability of.
   * @return Log probability of the given observation.
   */
  double LogProbability(const arma::vec& observation) const
  {
    // TODO: consider storing log probabilities instead?
    return log(Probability(observation));
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
   * Estimate the probability distribution directly from the given observations.
   * If any of the observations is greater than numObservations, a crash is
   * likely to occur.
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
   *    actually from this distribution.
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
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    // We serialize the vector manually since there seem to be some problems
    // with some boost versions.
    size_t dimensionality;
    dimensionality = probabilities.size();
    ar & data::CreateNVP(dimensionality, "dimensionality");

    if (Archive::is_loading::value)
    {
      probabilities.clear();
      probabilities.resize(dimensionality);
    }

    for (size_t i = 0; i < dimensionality; ++i)
    {
      std::ostringstream oss;
      oss << "probabilities" << i;
      ar & data::CreateNVP(probabilities[i], oss.str());
    }
  }

 private:
  //! The probabilities for each dimension; each arma::vec represents the
  //! probabilities for the observations in each dimension.
  std::vector<arma::vec> probabilities;
};

} // namespace distribution
} // namespace mlpack

#endif
