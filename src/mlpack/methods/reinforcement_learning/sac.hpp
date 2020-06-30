/**
 * @file sac.hpp
 * @author Nishant Kumar
 *
 * This file is the definition of SAC class, which implements the
 * soft actor-critic algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_SAC_HPP
#define MLPACK_METHODS_RL_SAC_HPP

#include <mlpack/prereqs.hpp>

#include "replay/random_replay.hpp"
#include "mlpack/methods/ann/activation_functions/tanh_function.hpp"
#include "mlpack/core/dists/gaussian_distribution.hpp"
#include "mlpack/methods/ann/dists/normal_distribution.hpp"
#include "training_config.hpp"

namespace mlpack {
namespace rl {

/**
 * TODO: Add citation
 *
 * @tparam EnvironmentType The environment of the reinforcement learning task.
 * @tparam NetworkType The network to compute action value.
 * @tparam UpdaterType How to apply gradients when training.
 * @tparam PolicyType Behavior policy of the agent.
 * @tparam ReplayType Experience replay method.
 */
template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename PolicyType,
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
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param config Hyper-parameters for training.
   * @param network The network to compute action value.
   * @param policy Behavior policy of the agent.
   * @param replayMethod Experience replay method.
   * @param updater How to apply gradients when training.
   * @param environment Reinforcement learning task.
   */
  SAC(TrainingConfig& config,
      QNetworkType& learningQ1Network,
      QNetworkType& learningQ2Network,
      PolicyNetworkType& policyNetwork,
      PolicyType& policy,
      ReplayType& replayMethod,
      UpdaterType qNetworkUpdater = UpdaterType(),
      UpdaterType policyNetworkUpdater = UpdaterType(),
      EnvironmentType environment = EnvironmentType());

  /**
    * Clean memory.
    */
  ~SAC();

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
   * Update the Q and policy networks.
   * */
  void Update();

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }
  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }


 private:
  //! Locally-stored hyper-parameters.
  TrainingConfig& config;

  //! Locally-stored learning Q1 and Q2 network.
  QNetworkType& learningQ1Network;
  QNetworkType& learningQ2Network;

  //! Locally-stored target Q1 and Q2 network.
  QNetworkType targetQ1Network;
  QNetworkType targetQ2Network;

  //! Locally-stored policy network.
  PolicyNetworkType& policyNetwork;

  //! Locally-stored behavior policy.
  PolicyType& policy;

  //! Locally-stored experience method.
  ReplayType& replayMethod;

  //! Locally-stored updater.
  UpdaterType qNetworkUpdater;
  #if ENS_VERSION_MAJOR >= 2
    typename UpdaterType::template
      Policy<arma::mat, arma::mat>* criticUpdatePolicy;
  #endif

  //! Locally-stored updater.
  UpdaterType policyNetworkUpdater;
  #if ENS_VERSION_MAJOR >= 2
    typename UpdaterType::template
    Policy<arma::mat, arma::mat>* actorUpdatePolicy;
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

} // namespace rl
} // namespace mlpack

// Include implementation
#include "sac_impl.hpp"
#endif
