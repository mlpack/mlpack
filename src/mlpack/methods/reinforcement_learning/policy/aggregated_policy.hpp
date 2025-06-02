/**
 * @file methods/reinforcement_learning/policy/aggregated_policy.hpp
 * @author Shangtong Zhang
 *
 * This file is the implementation of AggregatedPolicy class.
 * An aggregated policy will randomly select a child policy under a given
 * distribution at each time step.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_POLICY_AGGREGATED_POLICY_HPP
#define MLPACK_METHODS_RL_POLICY_AGGREGATED_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distributions/discrete_distribution.hpp>

namespace mlpack {

/**
 * @tparam PolicyType The type of the child policy.
 */
template <typename PolicyType>
class AggregatedPolicy
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename PolicyType::ActionType;

  /**
   * @param policies Child policies.
   * @param distribution Probability distribution for each child policy.
   *     User should make sure its size is same as the number of policies
   *     and the sum of its element is equal to 1.
   */
  AggregatedPolicy(std::vector<PolicyType> policies,
                   const arma::colvec& distribution) :
      policies(std::move(policies)),
      sampler({distribution})
  { /* Nothing to do here. */ };

  /**
   * Sample an action based on given action values.
   *
   * @param actionValue Values for each action.
   * @param deterministic Always select the action greedily.
   * @return Sampled action.
   */
  ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
  {
    if (deterministic)
      return policies.front().Sample(actionValue, true);
    size_t selected = arma::as_scalar(sampler.Random());
    return policies[selected].Sample(actionValue, false);
  }

  /**
   * Exploration probability will anneal at each step.
   */
  void Anneal()
  {
    for (PolicyType& policy : policies)
      policy.Anneal();
  }

 private:
  //! Locally-stored child policies.
  std::vector<PolicyType> policies;

  //! Locally-stored sampler under the given distribution.
  DiscreteDistribution<> sampler;
};

} // namespace mlpack

#endif
