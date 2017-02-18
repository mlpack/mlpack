/**
 * @file discrete_distribution.cpp
 * @author Ryan Curtin
 *
 * Implementation of DiscreteDistribution probability distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "discrete_distribution.hpp"

using namespace mlpack;
using namespace mlpack::distribution;

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
arma::vec DiscreteDistribution::Random() const
{
  size_t dimension = probabilities.size();
  arma::vec result(dimension);

  for (size_t d = 0; d < dimension; d++)
  {
    // Generate a random number.
    double randObs = math::Random();

    double sumProb = 0;

    for (size_t obs = 0; obs < probabilities[d].n_elem; obs++)
    {
      if ((sumProb += probabilities[d][obs]) >= randObs)
      {
        result[d] = obs;
        break;
      }
    }

    if (sumProb >= randObs != true)
      // This shouldn't happen.
      result[d] = probabilities[d].n_elem - 1;
  }
  return result;
}

/**
 * Estimate the probability distribution directly from the given observations.
 */
void DiscreteDistribution::Train(const arma::mat& observations)
{
  // Make sure the observations have same dimension with the probabilities
  if(observations.n_rows != probabilities.size())
  {
    Log::Debug << "the obversation must has the same dimension with the probabilities"
        << "the observation's dimension is" << observations.n_cols << "but the dimension of "
        << "probabilities is" << probabilities.size() << std::endl;
  }
  // Get the dimension size of the distribution
  const size_t dimensions = probabilities.size();

  // Iterate all the probabilities in each dimension
  for (size_t i=0; i < dimensions; i++)
  {
    // Clear the old probabilities
    probabilities[i].zeros();
    for (size_t r=0; r < observations.n_cols; r++)
      {
      // Add the probability of each observation.  The addition of 0.5 to the
      // observation is to turn the default flooring operation of the size_t cast
      // into a rounding observation.
      const size_t obs = size_t(observations(i, r) + 0.5);

      // Ensure that the observation is within the bounds.
      if (obs >= probabilities[i].n_elem)
      {
        Log::Debug << "DiscreteDistribution::Train(): observation " << i
            << " (" << obs << ") is invalid; observation must be in [0, "
            << probabilities[i].n_elem << "] for this distribution." << std::endl;       
      }
      probabilities[i][obs]++;
      }

    // Now normailze the distribution.
    double sum = accu(probabilities[i]);
    if (sum > 0)
      probabilities[i] /= sum;
    else
      probabilities[i].fill(1.0 / probabilities[i].n_elem); // Force normalization.
  }
}

/**
 * Estimate the probability distribution from the given observations when also
 * given probabilities that each observation is from this distribution.
 */
void DiscreteDistribution::Train(const arma::mat& observations,
                                    const arma::vec& probObs)
{
  // Make sure the observations have same dimension with the probabilities
  if(observations.n_rows != probabilities.size())
    {
      Log::Debug << "the obversation must has the same dimension with the probabilities"
          << "the observation's dimension is" << observations.n_rows<< "but the dimension of "
          << "probabilities is" << probabilities.size() << std::endl;
    }

  // Get the dimension size of the distribution
  size_t dimensions = probabilities.size();
  for (size_t i=0; i < dimensions; i++)
  {
    // Clear the old probabilities
    probabilities[i].zeros();

    // Ensure that the observation is within the bounds.
    for (size_t r=0; r < observations.n_cols; r++)
    {
      // Add the probability of each observation.  The addition of 0.5 to the
      // observation is to turn the default flooring operation of the size_t cast
      // into a rounding observation.

      const size_t obs = size_t(observations(i, r)+ 0.5);

      // Ensure that the observation is within the bounds.
      if (obs >= probabilities[i].n_elem)
      {
        Log::Debug << "DiscreteDistribution::Train(): observation " << i
            << " (" << obs << ") is invalid; observation must be in [0, "
            << probabilities[i].n_elem << "] for this distribution." << std::endl;       
      }

      probabilities[i][obs] += probObs[r];
    }

    // Now normailze the distribution.
    double sum = accu(probabilities[i]);
    if (sum > 0)
      probabilities[i] /= sum;
    else
      probabilities[i].fill(1.0 / probabilities[i].n_elem); // Force normalization.
  }
}
