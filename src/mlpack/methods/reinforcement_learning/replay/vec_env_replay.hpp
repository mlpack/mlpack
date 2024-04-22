/**
 * @file methods/reinforcement_learning/replay/vec_env_replay.hpp
 * @author Ali Hossam
 *
 * This file defines the `VecEnvReplay` class, which is a wrapper for managing
 * replay buffers in reinforcement learning (RL) with multiple environments.
 * It lets you store experiences from many RL environments in a replay buffer.
 *
 * mlpack is free software; you can use it and change it under the terms of the
 * 3-clause BSD license. For more details, check the license at:
 * http://www.opensource.org/licenses/BSD-3-Clause
 */

#ifndef MLPACK_METHODS_RL_REPLAY_VEC_ENV_REPLAY_HPP
#define MLPACK_METHODS_RL_REPLAY_VEC_ENV_REPLAY_HPP

#include "replay.hpp"
#include <mlpack/methods/reinforcement_learning/environment/vec_env.hpp>

namespace mlpack {

/**
 * @tparam EnvironmentType The type of RL environment.
 * @tparam ReplayType The type of replay buffer.
 *
 * The `VecEnvReplay` class wraps a replay buffer to work with vectorized RL
 * environments. It can handle experiences from multiple environments at once.
 */
template <typename EnvironmentType, typename ReplayType>
class VecEnvReplay
{
 private:
  /** The replay buffer we're wrapping. */
  ReplayType replay;

 public:
  /** Type for Environment Action. */
  using ActionType = typename EnvironmentType::Action;

  /** Type for Environment State. */
  using StateType = typename EnvironmentType::State;

  /** Type for vectorized state. */
  using VecStateType = typename VecEnv<EnvironmentType>::State;

  /** Type for vectorized action. */
  using VecActionType = typename VecEnv<EnvironmentType>::Action;

  /**
   * Constructor with a replay buffer.
   *
   * @param replay The replay buffer to use.
   */
  VecEnvReplay(ReplayType& replay) : replay(replay) {}

  /**
   * Store experiences from multiple environments.
   *
   * @param state The current states from the environments.
   * @param action The actions taken.
   * @param reward The rewards received.
   * @param nextState The next states.
   * @param isEnd Flags for whether the environments ended.
   * @param discount A discount factor for future rewards.
   */
  void Store(VecStateType state,
             VecActionType action,
             std::vector<double> reward,
             VecStateType nextState,
             std::vector<bool> isEnd,
             const double& discount)
  {
    for (size_t i = 0; i < state.Encode().size(); i++)
      replay.Store(state.Encode()[i], action.action[i], reward[i], nextState.Encode()[i], isEnd[i], discount);
  }

  /**
   * Get info from the replay buffer for n-step updates.
   *
   * @param reward The total reward.
   * @param nextState The state after n steps.
   * @param isEnd A flag for whether the episode ended.
   * @param discount A discount factor.
   */
  void GetNStepInfo(double& reward,
                    StateType& nextState,
                    bool& isEnd,
                    const double& discount)
  {
    replay.GetNStepInfo(reward, nextState, isEnd, discount);
  }

  /**
   * Get random samples from the replay buffer.
   *
   * @param sampledStates The sampled states.
   * @param sampledActions The sampled actions.
   * @param sampledRewards The sampled rewards.
   * @param sampledNextStates The sampled next states.
   * @param isTerminal A flag for terminal states.
   */
  void Sample(arma::mat& sampledStates,
              std::vector<ActionType>& sampledActions,
              arma::rowvec& sampledRewards,
              arma::mat& sampledNextStates,
              arma::irowvec& isTerminal)
  {
    replay.Sample(sampledStates, sampledActions, sampledRewards, sampledNextStates, isTerminal);
  }

  /**
   * Get the number of stored experiences.
   *
   * @return The number of stored experiences.
   */
  const size_t& Size()
  {
    return replay.Size();
  }

  /**
   * Update the replay buffer with new experiences.
   *
   * @param target The target values.
   * @param sampledActions The actions used for the update.
   * @param nextActionValues The next action values.
   * @param gradients The gradients for updating.
   */
  void Update(arma::mat target,
              std::vector<ActionType> sampledActions,
              arma::mat nextActionValues,
              arma::mat& gradients)
  {
    replay.Update(target, sampledActions, nextActionValues, gradients);
  }

  /**
   * Get the number of steps used for n-step updates.
   *
   * @return The number of steps.
   */
  const size_t& NSteps() const
  {
    return replay.NSteps();
  }
};

} // namespace mlpack

#endif
