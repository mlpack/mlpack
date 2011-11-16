/**
 * @file discrete_distribution.cpp
 * @author Ryan Curtin
 *
 * Implementation of DiscreteDistribution probability distribution.
 */
#include "discrete_distribution.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
size_t DiscreteDistribution::Random() const
{
  // Generate a random number.
  double randObs = (double) rand() / (double) RAND_MAX;

  double sumProb = 0;
  for (size_t obs = 0; obs < probabilities.n_elem; obs++)
    if ((sumProb += probabilities[obs]) >= randObs)
      return obs;

  // This shouldn't happen.
  return probabilities.n_elem - 1;
}

/**
 * Estimate the probability distribution directly from the given observations.
 */
void DiscreteDistribution::Estimate(const std::vector<size_t> observations)
{
  // Clear old probabilities.
  probabilities.zeros();

  // Add the probability of each observation.
  for (std::vector<size_t>::const_iterator it = observations.begin();
       it != observations.end(); it++)
    probabilities(*it)++;

  // Now normalize the distribution.
  double sum = accu(probabilities);
  if (sum > 0)
    probabilities /= sum;
  else
    probabilities.fill(1 / probabilities.n_elem); // Force normalization.
}

/**
 * Estimate the probability distribution from the given observations when also
 * given probabilities that each observation is from this distribution.
 */
void DiscreteDistribution::Estimate(const std::vector<size_t> observations,
                                    const std::vector<double> probObs)
{
  // Clear old probabilities.
  probabilities.zeros();

  // Add the probability of each observation.
  for (size_t i = 0; i < observations.size(); i++)
    probabilities(observations[i]) += probObs[i];

  // Now normalize the distribution.
  double sum = accu(probabilities);
  if (sum > 0)
    probabilities /= sum;
  else
    probabilities.fill(1 / probabilities.n_elem); // Force normalization.
}

/**
 * Set the vector of probabilities correctly.
 */
void DiscreteDistribution::Probabilities(const arma::vec& probabilities)
{
  double sum = accu(probabilities);
  if (sum > 0)
    this->probabilities = probabilities / sum;
  else
  {
    this->probabilities.set_size(probabilities.n_elem);
    this->probabilities.fill(1 / probabilities.n_elem);
  }
}
