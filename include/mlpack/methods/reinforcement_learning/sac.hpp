/**
 * @file methods/reinforcement_learning/sac.hpp
 * @author Nishant Kumar
 *
 * This file is the definition of SAC class, which implements the
 * Soft Actor-Critic algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_SAC_HPP
#define MLPACK_METHODS_RL_SAC_HPP

#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "replay/replay.hpp"
#include "training_config.hpp"

namespace mlpack {

/**
 * Implementation of Soft Actor-Critic, a model-free off-policy actor-critic
 * based deep reinforcement learning algorithm.
 *
 * For more details, see the following:
 * @code
 * @misc{haarnoja2018soft,
 *  author    = {Tuomas Haarnoja and
 *               Aurick Zhou and
 *               Kristian Hartikainen and
 *               George Tucker and
 *               Sehoon Ha and
 *               Jie Tan and
 *               Vikash Kumar and
 *               Henry Zhu and
 *               Abhishek Gupta and
 *               Pieter Abbeel and
 *               Sergey Levine},
 *  title     = {Soft Actor-Critic Algorithms and Applications},
 *  year      = {2018},
 *  url       = {https://arxiv.org/abs/1812.05905}
 * }
 * @endcode
 *
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam NetworkType The network to compute action value.
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
class SAC
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Create the SAC object with given settings.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, you can directly pass the parameter, as the constructor takes
   * a reference. This avoids unnecessary copy.
   *
   * @param config Hyper-parameters for training.
   * @param learningQ1Network The network to compute action value.
   * @param policyNetwork The network to produce an action given a state.
   * @param replayMethod Experience replay method.
   * @param qNetworkUpdater How to apply gradients to Q network when training.
   * @param policyNetworkUpdater How to apply gradients to policy network
   *        when training.
   * @param environment Reinforcement learning task.
   */
  SAC(TrainingConfig& config,
      QNetworkType& learningQ1Network,
      PolicyNetworkType& policyNetwork,
      ReplayType& replayMethod,
      UpdaterType qNetworkUpdater = UpdaterType(),
      UpdaterType policyNetworkUpdater = UpdaterType(),
      EnvironmentType environment = EnvironmentType());

  /**
    * Clean memory.
    */
  ~SAC();

  /**
   * Softly update the learning Q network parameters to the target Q network
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
#include "sac_impl.hpp"
#endif
