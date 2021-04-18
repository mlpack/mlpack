/**
 * @file methods/reinforcement_learning/ddpg.hpp
 * @author Tri Wahyu Guntara
 *
 * This file is the definition of DDPG class, which implements the
 * Deep Deterministic Policy Gradient algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_DDPG_HPP
#define MLPACK_METHODS_RL_DDPG_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

#include "training_config.hpp"
#include "replay/random_replay.hpp"

namespace mlpack {
namespace rl {

/**
 * @brief Implementation of Deep Deterministic Policy Gradient, a model-free
 * off-policy actor-critic based deep reinforcement learning algorithm for
 * continuous action spaces. DDPG can be thought of as being deep Q-learning
 * for continous action spaces.
 *
 * For more details, see the following:
 * @code
 * @misc{lillicrap2015continuous,
 *  author    = {Timothy P. Lillicrap and
 *               Jonathan J. Hunt and
 *               Alexander Pritzel and
 *               Nicolas Heess and
 *               Tom Erez and
 *               Yuval Tassa and
 *               David Silver and
 *               Daan Wierstra},
 *  title     = {Continuous control with deep reinforcement learning},
 *  year      = {2015},
 *  url       = {https://arxiv.org/abs/1509.02971}
 * }
 * @endcode
 *
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam QNetworkType The network to compute action value.
 * @tparam PolicyNetworkType The network to compute action given state.
 * @tparam UpdaterType How to apply gradients when training.
 * @tparam ReplayType Experience replay method.
 */
template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
class DDPG
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * @brief Create the DDPG object given settings.
   * 
   * @param config Hyper-parameter for training of type
   *        `mlpack::rl::TrainingConfig`.
   * @param qNetwork The network to compute action value.
   * @param policyNetwork The network to compute action given state.
   * @param replayMethod Experience replay method.
   * @param qNetworkUpdater How to apply gradients when training Q network.
   * @param policyNetworkUpdater How to apply gradients when training
   *        policy network
   * @param environment The environment of the reinforcement learning task.
   */
  DDPG(TrainingConfig& config,
       QNetworkType& qNetwork,
       PolicyNetworkType& policyNetwork,
       ReplayType& replayMethod,
       UpdaterType qNetworkUpdater = UpdaterType(),
       UpdaterType policyNetworkUpdater = UpdaterType(),
       EnvironmentType environment = EnvironmentType());
  
  /**
   * @brief Destroy the DDPG object and clean memory.
   */
  ~DDPG();
  
  /**
   * @brief Select action by passing state to policy network.
   */
  void SelectAction();

  /**
   * @brief Update the target networks by Polyak Averaging.
   * 
   * @param rho Weight of the current networks for averaging (0 <= rho <= 1).
   */
  void SoftUpdateTargetNetwork(double rho);

  /**
   * @brief Update the Q and policy networks.
   */
  void TrainAgent();
  
  /**
   * @brief Execute an episode.
   * 
   * @return Return of the episode.
   */
  double Episode();

  //! Modify the total steps from the beginning.
  size_t& TotalSteps() { return totalSteps; }
  //! Get the total steps from the beginning.
  const size_t& TotalSteps() const { return totalSteps; }

  //! Modify the state of the agent.
  StateType& State() { return state; }
  //! Get the state of the agent.
  const StateType& State() const { return state; }

  //! Get the action of the agent.
  const ActionType& Action() const { return action; }

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }
  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }

 private:
  //! Locally-stored hyper-parameters.
  TrainingConfig& config;

  //! Locally-stored actual and target policy network.
  PolicyNetworkType& policyNetwork;
  PolicyNetworkType targetPolicyNetwork;
  
  //! Locally-stored actual and target Q network.
  QNetworkType& qNetwork;
  QNetworkType targetQNetwork;
  
  //! Locally-stored experience replay method.
  ReplayType& replayMethod;

  //! Locally-stored updater for Q network.
  UpdaterType qNetworkUpdater;
  #if ENS_VERSION_MAJOR >= 2
  typename UpdaterType::template Policy<arma::mat, arma::mat>*
    qNetworkUpdatePolicy;
  #endif

  //! Locally-stored updater for policy network.
  UpdaterType policyNetworkUpdater;
  #if ENS_VERSION_MAJOR >= 2
  typename UpdaterType::template Policy<arma::mat, arma::mat>*
    policyNetworkUpdatePolicy;
  #endif

  //! Locally-stored environment of the reinforcement learning task.
  EnvironmentType environment;
  
  //! Locally-stored current state and action of the agent.
  StateType state;
  ActionType action;

  //! Locally-stored total steps from the beginning of the task.
  size_t totalSteps;

  //! Locally-stored flag indicating training mode / test mode.
  bool deterministic;

  //! Locally-stored loss function for Q network.
  mlpack::ann::MeanSquaredError<> lossFunction;
};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "ddpg_impl.hpp"
#endif