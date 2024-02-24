/**
 * @file methods/ann/dists/normal_distribution_impl.hpp
 * @author xiaohong ji
 * @author Nishant Kumar
 *
 * Implementation of the Normal Distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "normal_distribution.hpp"

namespace mlpack {

template<typename DataType>
NormalDistribution<DataType>::NormalDistribution()
{
  // Nothing to do here.
}

template<typename DataType>
NormalDistribution<DataType>::NormalDistribution(
    const DataType& mean,
    const DataType& sigma) :
    mean(mean),
    sigma(sigma)
{
  // Nothing to do here.
}

template<typename DataType>
DataType NormalDistribution<DataType>::Sample() const
{
  return sigma * randn<DataType>(mean.n_elem) + mean;
}

template<typename DataType>
DataType NormalDistribution<DataType>::LogProbability(
    const DataType& observation) const
{
  const DataType v1 = log(sigma) + std::log(std::sqrt(2 * M_PI));
  const DataType v2 = square(observation - mean) / (2 * square(sigma));
  return  (-v1 - v2);
}

template<typename DataType>
void NormalDistribution<DataType>::ProbBackward(
    const DataType& observation,
    DataType& dmu,
    DataType& dsigma) const
{
  dmu = (observation - mean) / (square(sigma)) % Probability(observation);
  dsigma = (-1.0 / sigma + (square(observation - mean) / pow(sigma, 3)))
            % Probability(observation);
}

template<typename DataType>
template<typename Archive>
void NormalDistribution<DataType>::serialize(Archive& ar,
                                             const uint32_t /* version */)
{
  ar(CEREAL_NVP(mean));
  ar(CEREAL_NVP(sigma));
}

} // namespace mlpack

#endif
