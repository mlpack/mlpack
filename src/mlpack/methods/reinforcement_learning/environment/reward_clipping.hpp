/**
 * @file methods/reinforcement_learning/environment/reward_clipping.hpp
 * @author Shashank Shekhar
 *
 * Reward clipping wrapper for RL environments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_REWARD_CLIPPING_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_REWARD_CLIPPING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Interface for clipping the reward to some value between the specified maximum and minimum value
 * (Clipping here is implemented as
 * \f$ g_{\text{clipped}} = \max(g_{\text{min}}, \min(g_{\text{min}}, g))) \f$.)
 *
 * @tparam EnvironmentType A type of Environment that is being wrapped.
 */

template <typename EnvironmentType>
class RewardClipping
{
 public:
  //! Convenient typedef for state.
  using State = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using Action = typename EnvironmentType::Action;

  /**
   * Constructor for creating a RewardClipping instance.
   *
   * @param minReward Minimum possible value of clipped reward.
   * @param maxReward Maximum possible value of clipped reward.
   * @param environment An instance of the environment used for actual
   *                    simulations.
   */
  RewardClipping(EnvironmentType& environment,
                 const double minReward = -1.0,
                 const double maxReward = 1.0) :
    environment(environment),
    minReward(minReward),
    maxReward(maxReward)
  {
    // Nothing to do here
  }

  /**
   * The InitialSample method is called by the environment to initialize the
   * starting state. Returns whatever Initial Sample is returned by the
   * environment.
   */
  State InitialSample()
  {
    return environment.InitialSample();
  }

  /**
   * Checks whether given state is a terminal state.
   * Returns the value by calling the environment method.
   *
   * @param state desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  {
    return environment.IsTerminal(state);
  }

  /**
   * Dynamics of Environment. The rewards returned from the base environment
   * are clipped according the maximum and minimum values specified.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return clippedReward, Reward clipped between [minReward, maxReward].
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Get original unclipped reward from base environment.
    double unclippedReward = environment.Sample(state, action, nextState);
    // Clip rewards according to the min and max limit and return.
    return std::min(std::max(unclippedReward, minReward), maxReward);
  }

  /**
   * Dynamics of Environment. The rewards returned from the base environment
   * are clipped according the maximum and minimum values specified.
   *
   * @param state The current state.
   * @param action The current action.
   * @return clippedReward, Reward clipped between [minReward, maxReward].
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  //! Get the environment.
  EnvironmentType& Environment() const { return environment; }
  //! Modify the environment.
  EnvironmentType& Environment() { return environment; }

  //! Get the minimum reward value.
  double MinReward() const { return minReward; }
  //! Modify the minimum reward value.
  double& MinReward() { return minReward; }

  //! Get the maximum reward value.
  double MaxReward() const { return maxReward; }
  //! Modify the maximum reward value.
  double& MaxReward() { return maxReward; }

 private:
  //! An instance of the UpdatePolicy used for actual optimization.
  EnvironmentType environment;

  //! Minimum possible value of clipped reward.
  double minReward;

  //! Maximum possible value of clipped reward.
  double maxReward;
};

} // namespace mlpack

#endif
