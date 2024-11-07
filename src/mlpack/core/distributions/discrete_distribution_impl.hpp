/**
 * @file core/distributions/discrete_distribution_impl.hpp
 * @author Ryan Curtin
 * @author Rohan Raj
 *
 * Implementation of DiscreteDistribution probability distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTRIBUTIONS_DISCRETE_DISTRIBUTION_IMPL_HPP

#include "discrete_distribution.hpp"

namespace mlpack {

/**
 * Return a randomly generated observation according to the probability
 * distribution defined by this object.
 */
template<typename MatType, typename ObsMatType>
inline typename DiscreteDistribution<MatType, ObsMatType>::ObsVecType
DiscreteDistribution<MatType, ObsMatType>::Random() const
{
  size_t dimension = probabilities.size();
  ObsVecType result(dimension);

  for (size_t d = 0; d < dimension; d++)
  {
    // Generate a random number.
    ElemType randObs = (ElemType) mlpack::Random();

    ElemType sumProb = 0;

    for (size_t obs = 0; obs < probabilities[d].n_elem; obs++)
    {
      if ((sumProb += probabilities[d][obs]) >= randObs)
      {
        result[d] = ObsType(obs);
        break;
      }
    }

    if (sumProb > 1.0)
    {
      // This shouldn't happen.
      result[d] = ObsType(probabilities[d].n_elem - 1);
    }
  }

  return result;
}

/**
 * Estimate the probability distribution directly from the given observations.
 */
template<typename MatType, typename ObsMatType>
inline void DiscreteDistribution<MatType, ObsMatType>::Train(
    const ObsMatType& observations)
{
  // Make sure the observations have same dimension as the probabilities.
  if (observations.n_rows != probabilities.size())
  {
    throw std::invalid_argument("observations must have same dimensionality as"
        " the DiscreteDistribution object");
  }

  // Get the dimension size of the distribution.
  const size_t dimensions = probabilities.size();

  // Clear the old probabilities.
  for (size_t i = 0; i < dimensions; ++i)
    probabilities[i].zeros();

  // Iterate over all the probabilities in each dimension.
  for (size_t r = 0; r < observations.n_cols; ++r)
  {
    for (size_t i = 0; i < dimensions; ++i)
    {
      // Add the probability of each observation.  The addition of 0.5 to the
      // observation is to turn the default flooring operation of the size_t
      // cast into a rounding observation.
      const size_t obs = (std::is_floating_point_v<ObsType>) ?
          size_t(observations(i, r) + 0.5) : size_t(observations(i, r));

      // Ensure that the observation is within the bounds.
      if (obs >= probabilities[i].n_elem)
      {
        std::ostringstream oss;
        oss << "observation " << r << " in dimension " << i << " ("
            << observations(i, r) << ") is invalid; must be in [0, "
            << probabilities[i].n_elem << "] for this distribution";
        throw std::invalid_argument(oss.str());
      }
      probabilities[i][obs]++;
    }
  }

  // Now normalize the distributions.
  for (size_t i = 0; i < dimensions; ++i)
  {
    ElemType sum = accu(probabilities[i]);
    if (sum > 0)
      probabilities[i] /= sum;
    else // Force normalization.
      probabilities[i].fill(1.0 / probabilities[i].n_elem);
  }
}

/**
 * Estimate the probability distribution from the given observations when also
 * given probabilities that each observation is from this distribution.
 */
template<typename MatType, typename ObsMatType>
inline void DiscreteDistribution<MatType, ObsMatType>::Train(
    const ObsMatType& observations,
    const VecType& probObs)
{
  // Make sure the observations have same dimension as the probabilities.
  if (observations.n_rows != probabilities.size())
  {
    throw std::invalid_argument("observations must have same dimensionality as"
        " the DiscreteDistribution object");
  }

  // Get the dimension size of the distribution.
  size_t dimensions = probabilities.size();

  // Clear the old probabilities.
  for (size_t i = 0; i < dimensions; ++i)
    probabilities[i].zeros();

  // Ensure that the observation is within the bounds.
  for (size_t r = 0; r < observations.n_cols; r++)
  {
    for (size_t i = 0; i < dimensions; ++i)
    {
      // Add the probability of each observation. The addition of 0.5
      // to the observation is to turn the default flooring operation
      // of the size_t cast into a rounding observation.
      const size_t obs = (std::is_floating_point_v<ObsType>) ?
          size_t(observations(i, r) + 0.5) : size_t(observations(i, r));

      // Ensure that the observation is within the bounds.
      if (obs >= probabilities[i].n_elem)
      {
        std::ostringstream oss;
        oss << "observation " << r << " in dimension " << i << " ("
            << observations(i, r) << ") is invalid; must be in [0, "
            << probabilities[i].n_elem << "] for this distribution";
        throw std::invalid_argument(oss.str());
      }

      probabilities[i][obs] += probObs[r];
    }
  }

  // Now normalize the distributions.
  for (size_t i = 0; i < dimensions; ++i)
  {
    ElemType sum = accu(probabilities[i]);
    if (sum > 0)
      probabilities[i] /= sum;
    else // Force normalization.
      probabilities[i].fill(1.0 / probabilities[i].n_elem);
  }
}

} // namespace mlpack

#endif
