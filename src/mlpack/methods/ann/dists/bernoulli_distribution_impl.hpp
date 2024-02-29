/**
 * @file methods/ann/dists/bernoulli_distribution_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Bernoulli distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "bernoulli_distribution.hpp"

namespace mlpack {

template<typename DataType>
BernoulliDistribution<DataType>::BernoulliDistribution() :
    applyLogistic(true),
    eps(1e-10)
{
  // Nothing to do here.
}

template<typename DataType>
BernoulliDistribution<DataType>::BernoulliDistribution(
    const DataType& param,
    const bool applyLogistic,
    const double eps) :
    logits(param),
    applyLogistic(applyLogistic),
    eps(eps)
{
  if (applyLogistic)
  {
    LogisticFunction::Fn(logits, probability);
  }
  else
  {
    probability = arma::mat(logits.memptr(), logits.n_rows,
        logits.n_cols, false, false);
  }
}

template<typename DataType>
DataType BernoulliDistribution<DataType>::Sample() const
{
  DataType sample;
  sample.randu(probability.n_rows, probability.n_cols);

  for (size_t i = 0; i < sample.n_elem; ++i)
      sample(i) = sample(i) < probability(i);

  return sample;
}

template<typename DataType>
double BernoulliDistribution<DataType>::LogProbability(
    const DataType& observation) const
{
  return accu(log(probability + eps) % observation +
      log(1 - probability + eps) % (1 - observation)) /
      observation.n_cols;
}

template<typename DataType>
void BernoulliDistribution<DataType>::LogProbBackward(
    const DataType& observation, DataType& output) const
{
  if (!applyLogistic)
  {
    output = observation / (probability + eps) - (1 - observation) /
        (1 - probability + eps);
  }
  else
  {
    LogisticFunction::Deriv(logits, probability, output);
    output = (observation / (probability + eps) - (1 - observation) /
        (1 - probability + eps)) % output;
  }
}

} // namespace mlpack

#endif
