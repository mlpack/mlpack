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
  // todo: sync the oldActorNetwork

  // Sample from previous experience.
  arma::mat sampledStates;
  std::vector<ActionType> sampledActions;
  arma::colvec sampledRewards;
  arma::mat sampledNextStates;
  arma::icolvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
      sampledNextStates, isTerminal);

  arma::colvec discountedRewards(sampledRewards.n_cols);
  arma::mat nextActionValues;
  double values = 0.0;
  criticNetwork.Predict(sampledNextStates, nextActionValues);
  for (size_t i = sampledRewards.n_cols; i > 0; --i)
  {
    values = sampledRewards[i - 1] + values * config.Discount();
    discountedRewards[sampledRewards.n_cols - i] = values;
  }

  arma::mat actionValues, advantages, criticGradients, actorGradients;
  criticNetwork.Predict(sampledStates, actionValues);

  advantages = discountedRewards - actionValues;

  // update the critic
  criticNetwork.Backward(advantages, criticGradients);
  updater.Update(criticNetwork.Parameters(), config.StepSize(),
      criticGradients);

  // update the actor
  arma::vec prob, oldProb;
  normalDist.Probability(actionValues, prob);
  oldNormalDist.Probability(actionValues, oldProb);
  arma::mat ratio =  prob / oldProb;

  arma::mat surrogateLoss = arma::clamp(ratio, 1 - config.Epsilon(),
      1 + config.Epsilon()) * advantages;
  arma::mat loss = - arma::min(ratio * advantages, surrogateLoss);

  actorNetwork.Backward(loss, actorGradients);
  updater.Update(actorNetwork.Parameters(), config.StepSize(), actorGradients);

  // update the oldActorNetwork
  oldActorNetwork = actorNetwork;
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
  arma::mat actionValue, sigma, mu;
  actorNetwork.Predict(state.Encode(), actionValue);

  ann::TanhFunction::Fn(actionValue.col(0), sigma);
  ann::SoftplusFunction::Fn(actionValue.col(1), mu);
  normalDist = distribution::GaussianDistribution(sigma, mu);

  oldActorNetwork.Predict(state.Encode(), actionValue);
  ann::TanhFunction::Fn(actionValue.col(0), sigma);
  ann::SoftplusFunction::Fn(actionValue.col(1), mu);
  oldNormalDist = distribution::GaussianDistribution(sigma, mu);

  ActionType action;
  action.action = normalDist.Random()[0];

  // Interact with the environment to advance to next state.
  StateType nextState;
  double reward = environment.Sample(state, action, nextState);

  // Store the transition for replay.
  replayMethod.Store(state, action, reward, nextState,
      environment.IsTerminal(nextState));

  // Update current state.
  state = nextState;

  if (deterministic)
    return reward;

  return reward;
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
