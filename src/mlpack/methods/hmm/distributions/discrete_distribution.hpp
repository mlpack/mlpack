/**
 * @file discrete_distribution.hpp
 * @author Ryan Curtin
 *
 * Implementation of the discrete distribution, where each discrete observation
 * has a given probability.
 */
#ifndef __MLPACK_METHODS_HMM_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP
#define __MLPACK_METHODS_HMM_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace distribution {

/**
 * A discrete distribution where the only observations are of type size_t.  This
 * is useful (for example) with discrete Hidden Markov Models, where
 * observations are non-negative integers representing specific emissions.
 *
 * No bounds checking is performed for observations, so if an invalid
 * observation is passed (i.e. observation > numObservations), a crash will
 * probably occur.
 */
class DiscreteDistribution
{
 public:
  //! The type of data which this distribution uses; in our case, non-negative
  //! integers represent observations.
  typedef size_t DataType;

  /**
   * Default constructor, which creates a distribution that has no observations.
   */
  DiscreteDistribution() { /* nothing to do */ }

  /**
   * Define the discrete distribution as having numObservations possible
   * observations.  The probability in each state will be set to (1 /
   * numObservations).
   *
   * @param numObservations Number of possible observations this distribution
   *    can have.
   */
  DiscreteDistribution(const size_t numObservations) :
      probabilities(arma::ones<arma::vec>(numObservations) / numObservations)
  { /* nothing to do */ }

  /**
   * Define the discrete distribution as having the given probabilities for each
   * observation.
   *
   * @param probabilities Probabilities of each possible observation.
   */
  DiscreteDistribution(const arma::vec& probabilities)
  {
    // We must be sure that our distribution is normalized.
    double sum = accu(probabilities);
    if (sum > 0)
      this->probabilities = probabilities / sum;
    else
    {
      this->probabilities.set_size(probabilities.n_elem);
      this->probabilities.fill(1 / probabilities.n_elem);
    }
  }

  /**
   * Return the probability of the given observation.  If the observation is
   * greater than the number of possible observations, then a crash will
   * probably occur -- bounds checking is not performed.
   *
   * @param observation Observation to return the probability of.
   * @return Probability of the given observation.
   */
  double Probability(size_t observation) const
  {
    return probabilities(observation);
  }

  /**
   * Return a randomly generated observation according to the probability
   * distribution defined by this object.
   *
   * @return Random observation.
   */
  size_t Random() const;

  /**
   * Estimate the probability distribution directly from the given observations.
   *
   * @param observations List of observations.
   */
  void Estimate(const std::vector<size_t> observations);

  /**
   * Estimate the probability distribution from the given observations, taking
   * into account the probability of each observation actually being from this
   * distribution.
   *
   * @param observations List of observations.
   * @param probabilities List of probabilities that each observation is
   *    actually from this distribution.
   */
  void Estimate(const std::vector<size_t> observations,
                const std::vector<double> probabilities);

  /**
   * Return the vector of probabilities.
   */
  const arma::vec& Probabilities() const { return probabilities; }

  /**
   * Set the vector of probabilities correctly.  The vector will be normalized.
   */
  void Probabilities(const arma::vec& probabilities);

 private:
  arma::vec probabilities;
};

}; // namespace distribution
}; // namespace mlpack

#endif
