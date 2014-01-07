/**
 * @file discrete_distribution.cpp
 * @author Ryan Curtin
 *
 * Implementation of DiscreteDistribution probability distribution.
 *
 * This file is part of MLPACK 1.0.8.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
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
  // Generate a random number.
  double randObs = math::Random();
  arma::vec result(1);

  double sumProb = 0;
  for (size_t obs = 0; obs < probabilities.n_elem; obs++)
  {
    if ((sumProb += probabilities[obs]) >= randObs)
    {
      result[0] = obs;
      return result;
    }
  }

  // This shouldn't happen.
  result[0] = probabilities.n_elem - 1;
  return result;
}

/**
 * Estimate the probability distribution directly from the given observations.
 */
void DiscreteDistribution::Estimate(const arma::mat& observations)
{
  // Clear old probabilities.
  probabilities.zeros();

  // Add the probability of each observation.  The addition of 0.5 to the
  // observation is to turn the default flooring operation of the size_t cast
  // into a rounding operation.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    const size_t obs = size_t(observations(0, i) + 0.5);

    // Ensure that the observation is within the bounds.
    if (obs >= probabilities.n_elem)
    {
      Log::Debug << "DiscreteDistribution::Estimate(): observation " << i
          << " (" << obs << ") is invalid; observation must be in [0, "
          << probabilities.n_elem << "] for this distribution." << std::endl;
    }

    probabilities(obs)++;
  }

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
void DiscreteDistribution::Estimate(const arma::mat& observations,
                                    const arma::vec& probObs)
{
  // Clear old probabilities.
  probabilities.zeros();

  // Add the probability of each observation.  The addition of 0.5 to the
  // observation is to turn the default flooring operation of the size_t cast
  // into a rounding observation.
  for (size_t i = 0; i < observations.n_cols; i++)
  {
    const size_t obs = size_t(observations(0, i) + 0.5);

    // Ensure that the observation is within the bounds.
    if (obs >= probabilities.n_elem)
    {
      Log::Debug << "DiscreteDistribution::Estimate(): observation " << i
          << " (" << obs << ") is invalid; observation must be in [0, "
          << probabilities.n_elem << "] for this distribution." << std::endl;
    }

    probabilities(obs) += probObs[i];
  }

  // Now normalize the distribution.
  double sum = accu(probabilities);
  if (sum > 0)
    probabilities /= sum;
  else
    probabilities.fill(1 / probabilities.n_elem); // Force normalization.
}

/*
 * Returns a string representation of this object.
 */
std::string DiscreteDistribution::ToString() const
{
  std::ostringstream convert;
  convert << "DiscreteDistribution [" << this << "]" << std::endl;
  convert << "Probabilities" << std::endl << probabilities;
  return convert.str();
}

