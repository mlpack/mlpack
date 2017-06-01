/**
 * @file q_learning.hpp
 * @author Shangtong Zhang
 *
 * This file is the definition of QLearning class,
 * which implements Q-Learning algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_Q_LEARNING_HPP
#define MLPACK_METHODS_RL_Q_LEARNING_HPP

#include <mlpack/prereqs.hpp>

#include "replay/random_replay.hpp"

namespace mlpack {
namespace rl {

/**
 * Implementation of various Q-Learning algorithms, such as DQN, double DQN.
 *
 * For more details, see the following:
 * @code
 * @article{Mnih2013,
 *  author    = {Volodymyr Mnih and
 *               Koray Kavukcuoglu and
 *               David Silver and
 *               Alex Graves and
 *               Ioannis Antonoglou and
 *               Daan Wierstra and
 *               Martin A. Riedmiller},
 *  title     = {Playing Atari with Deep Reinforcement Learning},
 *  journal   = {CoRR},
 *  year      = {2013},
 *  url       = {http://arxiv.org/abs/1312.5602}
 * }
 * @endcode
 *
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam NetworkType The network to compute action value.
 * @tparam OptimizerType The optimizer to train the network.
 * @tparam PolicyType Behavior policy of the agent.
 * @tparam ReplayType Experience replay method.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename OptimizerType,
  typename PolicyType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
class QLearning
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Create the QLearning object with given settings.
   *
   * @param network The network to compute action value.
   * @param optimizer The optimizer to train the network.
   * @param discount Discount for future return.
   * @param policy Behavior policy of the agent.
   * @param replayMethod Experience replay method.
   * @param targetNetworkSyncInterval Interval (steps) to sync the target network.
   * @param explorationSteps Steps before starting to learn.
   * @param doubleQLearning Whether to use double Q-Learning.
   * @param stepLimit Maximum steps in each episode, 0 means no limit.
   * @param environment Reinforcement learning task.
   */
  QLearning(NetworkType& network,
            OptimizerType& optimizer,
            double discount,
            PolicyType policy,
            ReplayType replayMethod,
            size_t targetNetworkSyncInterval,
            size_t explorationSteps,
            bool doubleQLearning = false,
            size_t stepLimit = 0,
            EnvironmentType environment = EnvironmentType());

  /**
   * Execute a step in an episode.
   * @return Reward for the step.
   */
  double Step();

  /**
   * Execute an episode.
   * @return Return of the episode.
   */
  double Episode();

  /**
   * @return Total steps from beginning.
   */
  const size_t& TotalSteps() const { return totalSteps; }

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }

  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }

 private:
  /**
   * Select the best action based on given action value.
   * @param actionValues Action values.
   * @return Selected actions.
   */
  arma::icolvec BestAction(const arma::mat& actionValues);

  //! Reference of the learning network.
  NetworkType& learningNetwork;

  //! Locally-stored target network.
  NetworkType targetNetwork;

  //! Reference of the optimizer.
  OptimizerType& optimizer;

  //! Discount factor of future return.
  double discount;

  //! Locally-stored behavior policy.
  PolicyType policy;

  //! Locally-stored experience method.
  ReplayType replayMethod;

  //! Interval (steps) to update target network.
  size_t targetNetworkSyncInterval;

  //! Random steps before starting to learn.
  size_t explorationSteps;

  //! Whether to use double Q-Learning.
  bool doubleQLearning;

  //! Maximum steps for each episode.
  size_t stepLimit;

  //! Locally-stored reinforcement learning task.
  EnvironmentType environment;

  //! Total steps from the beginning of the task.
  size_t totalSteps;

  //! Locally-stored current state of the agent.
  StateType state;

  //! Locally-stored flag indicating training mode or test mode.
  bool deterministic;
};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "q_learning_impl.hpp"
#endif