/**
 * @file stochiastic_policy.hpp
 * @author Rohan Raj
 *
 * This file is an implementation of action selection from stochaistic policy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_POLICY_STOCHIASTIC_POLICY_HPP
#define MLPACK_METHODS_RL_POLICY_STOCHIASTIC_POLICY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation for stochiastic policy.
 *
 * In general we will select an action based on the action value.
 * This means that the result of actor networks will be used to
 * determine the action to be taken.
 *
 * @tparam EnvironmentType The reinforcement learning task.
 */
template <typename EnvironmentType>
class StochiasticPolicy
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;
  /**
   * Sample an action based on given action values probability.
   *
   * @param actionValue probaility for each action.
   * @param deterministic Always select the action greedily on probability.
   * @return Sampled action.
   */
  ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
  {
  size_t actionLength = static_cast<size_t>(ActionType::size);
  ActionType action = static_cast<ActionType>(math::RandInt(ActionType::size));
  double probability = 0.0;
  double randomProbability = math::Random(0.0, 1.0);
  if (!deterministic)
  {
    for (size_t i = 0; i < actionLength; ++i)
    {
      probability += actionValue(i);
      if (probability >= randomProbability || i == actionLength - 1)
      {
        action = static_cast<ActionType>(i);
        break;
      }
    }
  }
  else
    action = static_cast<ActionType>(
        arma::as_scalar(arma::find(actionValue == actionValue.max(), 1)));

  return action;
  }
};

} // namespace rl
} // namespace mlpack

#endif
