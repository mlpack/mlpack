/**
 * @file methods/reinforcement_learning/td3.hpp
 * @author Tarek Elsayed
 *
 * This file is the definition of TD3 class, which implements the
 * Twin Delayed Deep Deterministic Policy Gradient algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_TD3_HPP
#define MLPACK_METHODS_RL_TD3_HPP

#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "replay/replay.hpp"
#include "training_config.hpp"

namespace mlpack {

/**
 * Implementation of Twin Delayed Deep Deterministic policy gradient, 
 * a model-free off-policy actor-critic based algorithm.
 *
 * For more details, see the following:
 * @code
 * @misc{Fujimoto et al, 2018,
 *  author    = {Scott Fujimoto,
 *               Herke van Hoof,
 *               David Meger},
 *  title     = {Addressing Function Approximation Error in 
 *               Actor-Critic Methods},
 *  year      = {2018},
 *  url       = {https://arxiv.org/abs/1802.09477}
 * }
 * @endcode
 *
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam QNetworkType The network used to estimate the critic's Q-values.
 * @tparam PolicyNetworkType The network to compute action value.
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
class TD3
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Create the TD3 object with given settings.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, you can directly pass the parameter, as the constructor takes
   * a reference. This avoids unnecessary copy.
   *
   * @param config Hyper-parameters for training.
   * @param learningQNetwork The network to compute action value.
   * @param policyNetwork The network to produce an action given a state.
   * @param replayMethod Experience replay method.
   * @param qNetworkUpdater How to apply gradients to Q network when training.
   * @param policyNetworkUpdater How to apply gradients to policy network
   *        when training.
   * @param environment Reinforcement learning task.
   */
  TD3(TrainingConfig& config,
      QNetworkType& learningQNetwork,
      PolicyNetworkType& policyNetwork,
      ReplayType& replayMethod,
      UpdaterType qNetworkUpdater = UpdaterType(),
      UpdaterType policyNetworkUpdater = UpdaterType(),
      EnvironmentType environment = EnvironmentType());

  /**
    * Clean memory.
    */
  ~TD3();

  /**
   * Softly update the target networks` parameters from the learning networks`
   * parameters.
   * 
   * @param rho How "softly" should the parameters be copied.
   * */
  void SoftUpdate(double rho);

  /**
   * Update the Q and policy networks.
   * */
  void Update();

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

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }
  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }


 private:
  //! Locally-stored hyper-parameters.
  TrainingConfig& config;

  //! Locally-stored learning Q1 and Q2 network.
  QNetworkType& learningQ1Network;
  QNetworkType learningQ2Network;

  //! Locally-stored target Q1 and Q2 network.
  QNetworkType targetQ1Network;
  QNetworkType targetQ2Network;

  //! Locally-stored policy network.
  PolicyNetworkType& policyNetwork;

  //! Locally-stored target policy network.
  PolicyNetworkType targetPNetwork;

  //! Locally-stored experience method.
  ReplayType& replayMethod;

  //! Locally-stored updater.
  UpdaterType qNetworkUpdater;
  #if ENS_VERSION_MAJOR >= 2
  typename UpdaterType::template Policy<arma::mat, arma::mat>*
      qNetworkUpdatePolicy;
  #endif

  //! Locally-stored updater.
  UpdaterType policyNetworkUpdater;
  #if ENS_VERSION_MAJOR >= 2
  typename UpdaterType::template Policy<arma::mat, arma::mat>*
      policyNetworkUpdatePolicy;
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

  //! Locally-stored loss function.
  MeanSquaredError lossFunction;
};

} // namespace mlpack

// Include implementation
#include "td3_impl.hpp"
#endif
