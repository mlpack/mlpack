/**
 * @file exp_decaay_greedy_policy.hpp
 * @author Shangtong Zhang
 * @author Abhinav Sagar
 * @author Arsen Zahray
 *
 * This file is an implementation of epsilon greedy policy with exponential decay.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_POLICY_EXP_DECAY_GREEDY_POLICY_HPP
#define MLPACK_METHODS_RL_POLICY_EXP_DECAY_GREEDY_POLICY_HPP

#include <mlpack/prereqs.hpp>
#include <type_traits>

namespace mlpack {
namespace rl {

/**
 * Implementation for epsilon greedy policy.
 *
 * In general we will select an action greedily based on the action value,
 * however sometimes we will also randomly select an action to encourage
 * exploration.
 *
 * @tparam EnvironmentType The reinforcement learning task.
 */
template <typename EnvironmentType>
class ExponentiallyDecayingGreedyPolicy
{
 public:
  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Constructor for epsilon greedy policy class.
   *
   * @param initialEpsilon The initial probability to explore
   *        (select a random action).
   * @param minEpsilon Epsilon will never be less than this value.
   * @param decayRate at each step, probability of selecting random
   *        action will decrease by `1-decayRate`
   */
  ExponentiallyDecayingGreedyPolicy(const double initialEpsilon,
               const double minEpsilon,
               const double decayRate = 1e-10) :
      epsilon(initialEpsilon),
      minEpsilon(minEpsilon),
      delta(1-decayRate)
  {
      // We are using static_cast in the Sample method.
      // This means, we have to check at compile time that the ActionType
      // can be used in this maner
      static_assert(std::is_enum<ActionType>::value,
              "ActionType must be an enum type. For non-enum types, use"
              " Cuntinuous policy");
  }

  /**
   * Sample an action based on given action values.
   *
   * @param actionValue Values for each action.
   * @param deterministic Always select the action greedily.
   * @return Sampled action.
   */
  ActionType Sample(const arma::colvec& actionValue, bool deterministic = false)
  {
    // Select the action randomly.
    if (!deterministic )
    {
        double exploration = math::Random();
        if (exploration < epsilon)
        {
            return static_cast<ActionType>(math::RandInt(ActionType::size));
        }
    }

    // Select the action greedily.
    return static_cast<ActionType>(
        arma::as_scalar(arma::find(actionValue == actionValue.max(), 1)));
  }

  /**
   * Exploration probability will anneal at each step.
   */
  void Anneal()
  {
    epsilon *= delta;
    epsilon = std::max(minEpsilon, epsilon);
  }

  /**
   * @return Current possibility to explore.
   */
  const double& Epsilon() const { return epsilon; }

 private:
  //! Locally-stored probability to explore.
  double epsilon;

  //! Locally-stored lower bound for epsilon.
  double minEpsilon;

  //! Locally-stored stride for epsilon to anneal.
  double delta;
};

} // namespace rl
} // namespace mlpack

#endif
