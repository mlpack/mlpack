/**
 * @file methods/reinforcement_learning/eql.hpp
 * @author Nanubala Gnana Sai
 *
 * This file is implements a Multi-objective Q learning algorithm,
 * which is called Envelope Q-learning.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_EQL_HPP
#define MLPACK_METHODS_RL_EQL_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

#include "replay/random_replay.hpp"
#include "replay/prioritized_replay.hpp"
#include "training_config.hpp"

namespace mlpack {
namespace rl {

/**
 * Implementation of EQL.
 *
 * For more details, see the following:
 * @code
 * @article{yang2019generalized,
 *  author    = {Yang and
 *               Runzhe and
 *               Sun and
 *               Xingyuan and
 *               Narasimhan and
 *               Karthik},
 *  title     = {A generalized algorithm for multi-objective reinforcement learning and policy adaptation},
 *  journal   = {arXiv preprint arXiv:1908.08342},
 *  year      = {2019},
 *  url       = {https://arxiv.org/abs/1908.08342}
 * }
 * @endcode
 *
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam NetworkType The network to compute utility value.
 * @tparam UpdaterType How to apply gradients when training.
 * @tparam PolicyType Behavior policy of the agent.
 * @tparam ReplayType Experience replay method.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType = PrioritizedReplay<EnvironmentType>
>
class EQL
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Create the EQL object with given settings.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param config Hyper-parameters for training.
   * @param network The network to compute utility value.
   * @param policy Behavior policy of the agent.
   * @param replayMethod Experience replay method.
   * @param updater How to apply gradients when training.
   * @param environment Reinforcement learning task.
   */
  EQL(TrainingConfig& config,
      NetworkType& network,
      PolicyType& policy,
      ReplayType& replayMethod,
      UpdaterType updater = UpdaterType(),
      EnvironmentType environment = EnvironmentType());

  /**
   * Clean memory.
   */
  ~EQL();

  /**
   * Trains the DQN agent.
   */
  void TrainAgent();

  /**
   * Select an action, given an agent.
   */
  void SelectAction();

  /**
   * Execute an episode.
   * @return Return of the episode.
   */
  arma::vec Episode();

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

  //! Modify the preference of the agent.
  arma::vec& Preference() { return preference; }
  //! Get the preference of the agent.
  const arma::vec& Preference() const { return preference; }

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
  arma::uvec BestAction(const arma::mat& actionValues);

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

  //! Locally-stored current preference of the agent.
  arma::vec preference;

  //! Locally-stored flag indicating training mode or test mode.
  bool deterministic;
};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "eql_impl.hpp"
#endif
