/**
 * @file normal_distribution_impl.hpp
 * @author xiaohong ji
 *
 * Implementation of the Normal distribution class.
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
namespace ann /** Artificial Neural Network. */ {

NormalDistribution::NormalDistribution()
{
  // Nothing to do here.
}

NormalDistribution::NormalDistribution(
    const arma::vec& mean,
    const arma::vec& sigma) :
    mean(mean),
    sigma(sigma)
{
}

arma::vec NormalDistribution::Sample() const
{
  return sigma * arma::randn<arma::vec>(mean.n_elem) + mean;
}

arma::vec NormalDistribution::LogProbability(
    const arma::vec& observation) const
{
  const arma::vec variance = arma::square(sigma);
  arma::vec v1 = sigma + std::log(std::sqrt(2 * pi));
  arma::vec v2 = arma::square(observation - sigma) / (2 * variance);
  return  (-v1 - v2);
}

} // namespace ann
} // namespace mlpack

#endif
