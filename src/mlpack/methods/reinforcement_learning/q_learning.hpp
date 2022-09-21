/**
 * @file methods/reinforcement_learning/q_learning.hpp
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

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "replay/replay.hpp"
#include "training_config.hpp"

namespace mlpack {

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
 * @tparam UpdaterType How to apply gradients when training.
 * @tparam PolicyType Behavior policy of the agent.
 * @tparam ReplayType Experience replay method.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
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
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param config Hyper-parameters for training.
   * @param network The network to compute action value.
   * @param policy Behavior policy of the agent.
   * @param replayMethod Experience replay method.
   * @param updater How to apply gradients when training.
   * @param environment Reinforcement learning task.
   */
  QLearning(TrainingConfig& config,
            NetworkType& network,
            PolicyType& policy,
            ReplayType& replayMethod,
            UpdaterType updater = UpdaterType(),
            EnvironmentType environment = EnvironmentType());

  /**
   * Clean memory.
   */
  ~QLearning();

  /**
   * Trains the DQN agent(non-categorical).
   */
  void TrainAgent();

  /**
   * Trains the DQN agent of categorical type.
   */
  void TrainCategoricalAgent();

  /**
   * Select an action, given an agent.
   */
  void SelectAction();

  /**
   * Execute an episode.
   * @return Return of the episode.
   */
  double Episode();

  //! Modify total steps from beginning.
  size_t& TotalSteps() { return totalSteps; }
  //! Get total steps from beginning.
  const size_t& TotalSteps() const { return totalSteps; }

  //! Modify the state of the agent.
  StateType& State() { return state; }
  //! Get the state of the agent.
  const StateType& State() const { return state; }

  //! Get the action of the agent.
  const ActionType& Action() const { return action; }

  //! Modify the environment in which the agent is.
  EnvironmentType& Environment() { return environment; }
  //! Get the environment in which the agent is.
  const EnvironmentType& Environment() const { return environment; }

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }
  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }

  //! Return the learning network.
  const NetworkType& Network() const { return learningNetwork; }
  //! Modify the learning network.
  NetworkType& Network() { return learningNetwork; }

 private:
  /**
   * Select the best action based on given action value.
   * @param actionValues Action values.
   * @return Selected actions.
   */
  arma::Col<size_t> BestAction(const arma::mat& actionValues);

  //! Locally-stored hyper-parameters.
  TrainingConfig& config;

  //! Locally-stored learning network.
  NetworkType& learningNetwork;

  //! Locally-stored target network.
  NetworkType targetNetwork;

  //! Locally-stored behavior policy.
  PolicyType& policy;

  //! Locally-stored experience method.
  ReplayType& replayMethod;

  //! Locally-stored updater.
  UpdaterType updater;
  #if ENS_VERSION_MAJOR >= 2
  typename UpdaterType::template Policy<arma::mat, arma::mat>* updatePolicy;
  #endif

  //! Locally-stored reinforcement learning task.
  EnvironmentType environment;

  //! Total steps from the beginning of the task.
  size_t totalSteps;

  //! Locally-stored current state of the agent.
  StateType state;

  //! Locally-stored action of the agent.
  ActionType action;

  //! Locally-stored flag indicating training mode or test mode.
  bool deterministic;
};

} // namespace mlpack

// Include implementation
#include "q_learning_impl.hpp"
#endif
