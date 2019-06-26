/**
 * @file ppo.hpp
 * @author Xiaohong Ji
 *
 * This file is the implementation of PPO class, which implements
 * proximal policy optimization algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_PPO_IMPL_HPP
#define MLPACK_METHODS_RL_PPO_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "ppo.hpp"

namespace mlpack {
namespace rl {

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::PPO(TrainingConfig config,
       ActorNetworkType actorNetwork,
       CriticNetworkType criticNetwork,
       PolicyType policy,
       ReplayType replayMethod,
       UpdaterType updater,
       EnvironmentType environment):
  config(std::move(config)),
  actorNetwork(std::move(actorNetwork)),
  criticNetwork(std::move(criticNetwork)),
  updater(std::move(updater)),
  policy(std::move(policy)),
  replayMethod(std::move(replayMethod)),
  environment(std::move(environment))
{
  if (actorNetwork.Parameters().is_empty())
    actorNetwork.ResetParameters();
  if (criticNetwork.Parameters().is_empty())
    criticNetwork.ResetParameters();

  this->updater.Initialize(actorNetwork.Parameters().n_rows,
                           actorNetwork.Parameters().n_cols);
  this->oldActorNetwork = actorNetwork;
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
void PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Update()
{
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Step()
{
  // Get the action value for each action at current state.
  arma::mat actionValue;
  actorNetwork.Predict(state.Encode(), actionValue);

  ann::TanhFunction tanh;
  ann::SoftplusFunction softp;

  arma::mat sigma, mu;
  tanh.Fn(actionValue.col(0), sigma);
  softp.Fn(actionValue.col(0), mu);

  distribution::GaussianDistribution normalDist =
    distribution::GaussianDistribution(sigma, mu);

  // Get the action value for each action at current state.
  actorNetwork.Predict(state.Encode(), actionValue);

  oldActorNetwork.Predict(state.Encode(), actionValue);
  tanh.Fn(actionValue.col(0), sigma);
  softp.Fn(actionValue.col(0), mu);

  distribution::GaussianDistribution oldnormalDist =
    distribution::GaussianDistribution(sigma, mu);

  ActionType action;
  action.action = normalDist.Random()[0];

  // Interact with the environment to advance to next state.
  StateType nextState;
  double reward = environment.Sample(state, action, nextState);

  // Store the transition for replay.
  replayMethod.Store(state, action, reward, nextState, 0);

  // Update current state.
  state = nextState;

  if (deterministic)
    return reward;

  return 0.0;
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Episode()
{
  // Get the initial state from environment.
  state = environment.InitialSample();

  // Track the steps in this episode.
  size_t steps = 0;

  // Track the return of this episode.
  double totalReturn = 0.0;

  // Running until get to the terminal state.
  while (config.StepLimit() && steps >= config.StepLimit())
  {
    totalReturn += Step();
    steps++;

    if (totalSteps % config.UpdateInterval() == 0)
    {
      Update();
    }

    totalSteps++;
  }
  return totalReturn;
}

} // namespace rl
} // namespace mlpack
#endif
