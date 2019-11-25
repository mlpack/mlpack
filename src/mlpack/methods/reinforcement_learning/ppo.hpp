/**
 * @file ppo.hpp
 * @author Xiaohong Ji
 *
 * This file is the definition of PPO class, which implements
 * proximal policy optimization algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_PPO_HPP
#define MLPACK_METHODS_RL_PPO_HPP

#include <mlpack/prereqs.hpp>

#include "replay/random_replay.hpp"
#include "mlpack/methods/ann/activation_functions/tanh_function.hpp"
#include "mlpack/core/dists/gaussian_distribution.hpp"
#include "mlpack/methods/ann/dists/normal_distribution.hpp"
#include "training_config.hpp"

namespace mlpack {
namespace rl {

/**
 * Implementation of PPO algorithms. PPO is a new family of policy gradient
 * methods for reinforcement learning, which alternate between sampling data
 * through interaction with the environment, and optimizing a 'surrogate'
 * objective function using stochastic gradient ascent.
 *
 * For more details, see the following:
 * @code
 * @article{Mnih2013,
 *  author    = {John Schulman and
 *               Filip Wolski and
 *               Prafulla Dhariwal and
 *               Alec Radford and
 *               Oleg Klimov},
 *  title     = {Proximal policy optimization algorithms},
 *  journal   = {arXiv preprint arXiv:1707.06347},
 *  year      = {2017},
 *  url       = {https://arxiv.org/abs/1707.06347}
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
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
class PPO
{
 public:
  //! Convenient typedef for state.
  using StateType = typename EnvironmentType::State;

  //! Convenient typedef for action.
  using ActionType = typename EnvironmentType::Action;

  /**
   * Create the PPO object with given settings.
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
  PPO(TrainingConfig config,
      ActorNetworkType ActorNetwork,
      CriticNetworkType CriticNetwork,
      PolicyType policy,
      ReplayType replayMethod,
      UpdaterType updater = UpdaterType(),
      EnvironmentType environment = EnvironmentType());

  /**
    * Clean memory.
    */
  ~PPO();

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
   * Update the actor and critic model
   * */
  void Update();

  //! Modify the training mode / test mode indicator.
  bool& Deterministic() { return deterministic; }
  //! Get the indicator of training mode / test mode.
  const bool& Deterministic() const { return deterministic; }


 private:
  //! Locally-stored hyper-parameters.
  TrainingConfig config;

  //! Locally-stored actor network.
  ActorNetworkType actorNetwork;

  //! Locally-stored old actor network.
  ActorNetworkType oldActorNetwork;

  //! Locally-stored critic network.
  CriticNetworkType criticNetwork;

  //! Locally-stored updater.
  UpdaterType criticUpdater;
  #if ENS_VERSION_MAJOR >= 2
    typename UpdaterType::template
      Policy<arma::mat, arma::mat>* criticUpdatePolicy;
  #endif

  //! Locally-stored updater.
  UpdaterType actorUpdater;
  #if ENS_VERSION_MAJOR >= 2
    typename UpdaterType::template
    Policy<arma::mat, arma::mat>* actorUpdatePolicy;
  #endif

  //! Locally-stored behavior policy.
  PolicyType policy;

  //! Locally-stored experience method.
  ReplayType replayMethod;

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
#include "ppo_impl.hpp"
#endif
