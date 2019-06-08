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
  typename NetworkType,
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
      NetworkType actor,
      NetworkType critic,
      PolicyType policy,
      ReplayType replayMethod,
      UpdaterType updater = UpdaterType(),
      EnvironmentType environment = EnvironmentType());


 private:
  //! Locally-stored hyper-parameters.
  TrainingConfig config;

  //! Locally-stored actor network.
  NetworkType ActorNetwork;

  //! Locally-stored critic network.
  NetworkType CriticNetwork;

};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "ppo_impl.hpp"
#endif
