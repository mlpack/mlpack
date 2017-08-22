/**
 * @file sample_policy.hpp
 * @author Shangtong Zhang
 *
 * This file is an implementation of sample policy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_POLICY_SAMPLE_POLICY_HPP
#define MLPACK_METHODS_RL_POLICY_SAMPLE_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation for sample policy.
 *
 * It will sample an action according to the given probability distribution.
 *
 * @tparam EnvironmentType The reinforcement learning task.
 */
template <typename EnvironmentType>
class SamplePolicy
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Sample an action based on given probability distribution.
   *
   * @param actionProb Desired probability distribution.
   * @param deterministic Always select the action with highest probability.
   * @return Sampled action.
   */
  ActionType Sample(const arma::colvec& actionProb, bool deterministic = false)
  {
    if (!deterministic)
      return static_cast<ActionType>(
          arma::as_scalar(arma::find(actionProb == actionProb.max(), 1)));

    distribution::DiscreteDistribution sampler({actionProb});
    return static_cast<ActionType >(arma::as_scalar(sampler.Random()));
  }
};

} // namespace rl
} // namespace mlpack

#endif
