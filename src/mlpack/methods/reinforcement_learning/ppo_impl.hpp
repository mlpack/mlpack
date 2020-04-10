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
       ActorNetworkType actor,
       CriticNetworkType critic,
       PolicyType policy,
       ReplayType replayMethod,
       UpdaterType updater,
       EnvironmentType environment):
  config(std::move(config)),
  actorNetwork(std::move(actor)),
  criticNetwork(std::move(critic)),
  actorUpdater(std::move(updater)),
  #if ENS_VERSION_MAJOR >= 2
  actorUpdatePolicy(NULL),
  #endif
  criticUpdater(std::move(updater)),
  #if ENS_VERSION_MAJOR >= 2
  criticUpdatePolicy(NULL),
  #endif
  policy(std::move(policy)),
  replayMethod(std::move(replayMethod)),
  environment(std::move(environment)),
  totalSteps(0),
  deterministic(false)
{
  // Set up actor and critic network.
  if (actorNetwork.Parameters().is_empty())
    actorNetwork.ResetParameters();

  if (criticNetwork.Parameters().is_empty())
    criticNetwork.ResetParameters();

  #if ENS_VERSION_MAJOR == 1
  this->criticUpdater.Initialize(criticNetwork.Parameters().n_rows,
                                 criticNetwork.Parameters().n_cols);
  #else
  this->criticUpdatePolicy = new typename UpdaterType::template
  Policy<arma::mat, arma::mat>(this->criticUpdater,
                               criticNetwork.Parameters().n_rows,
                               criticNetwork.Parameters().n_cols);
  #endif

  #if ENS_VERSION_MAJOR == 1
  this->actorUpdater.Initialize(actorNetwork.Parameters().n_rows,
                                actorNetwork.Parameters().n_cols);
  #else
  this->actorUpdatePolicy = new typename UpdaterType::template
  Policy<arma::mat, arma::mat>(this->actorUpdater,
                               actorNetwork.Parameters().n_rows,
                               actorNetwork.Parameters().n_cols);
  #endif

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
PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::~PPO()
{
  #if ENS_VERSION_MAJOR >= 2
  delete actorUpdatePolicy;
  delete criticUpdatePolicy;
  #endif
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
  arma::mat sampledStates;
  std::vector<ActionType> sampledActions;
  arma::colvec sampledRewards;
  arma::mat sampledNextStates;
  arma::icolvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
                      sampledNextStates, isTerminal);

  arma::rowvec discountedRewards(sampledRewards.n_rows);
  arma::mat nextActionValues;
  double values = 0.0;
  criticNetwork.Predict(sampledNextStates, nextActionValues);
  for (size_t i = sampledRewards.n_cols; i > 0; --i)
  {
    values = sampledRewards[i - 1] + values * config.Discount();
    discountedRewards[sampledRewards.n_cols - i] = values;
  }

  arma::mat actionValues, advantages, criticGradients, actorGradients;
  criticNetwork.Forward(sampledStates, actionValues);

  advantages = arma::conv_to<arma::mat>::
               from(discountedRewards) - actionValues;

  // Update the critic.
  criticNetwork.Backward(sampledStates, advantages, criticGradients);
  #if ENS_VERSION_MAJOR == 1
  criticUpdater.Update(criticNetwork.Parameters(), config.StepSize(),
                       criticGradients);
  #else
  criticUpdatePolicy->Update(criticNetwork.Parameters(), config.StepSize(),
                             criticGradients);
  #endif

  for (size_t step = 0; step < config.ActorUpdateStep(); step ++) {
    // calculate the ratio.
    arma::mat actionParameter, sigma, mu;
    actorNetwork.Forward(sampledStates, actionParameter);

    ann::TanhFunction::Fn(actionParameter.row(0), mu);
    ann::SoftplusFunction::Fn(actionParameter.row(1), sigma);

    ann::NormalDistribution normalDist =
      ann::NormalDistribution(vectorise(mu, 0), vectorise(sigma, 0));

    oldActorNetwork.Forward(sampledStates, actionParameter);
    ann::TanhFunction::Fn(actionParameter.row(0), mu);
    ann::SoftplusFunction::Fn(actionParameter.row(1), sigma);

    ann::NormalDistribution oldNormalDist =
      ann::NormalDistribution(vectorise(mu, 0), vectorise(sigma, 0));

    // Update the actor.
    // observation use action.
    arma::vec prob, oldProb;
    arma::colvec observation(sampledActions.size());
    for (size_t i = 0; i < sampledActions.size(); i++)
    {
      observation[i] = sampledActions[i].action[0];
    }
    normalDist.LogProbability(observation, prob);
    oldNormalDist.LogProbability(observation, oldProb);

    arma::mat ratio = arma::exp((prob - oldProb).t());

    arma::mat L1 = ratio % advantages;
    arma::mat L2 = arma::clamp(ratio, 1 - config.Epsilon(),
                              1 + config.Epsilon()) % advantages;
    arma::mat surroLoss = -arma::min(L1, L2);

    // Calculates the gradient for Surrogate Loss
    arma::mat dL1 = (L1 < L2) % advantages;
    arma::mat dL2 = (L1 >= L2) % (ratio >= (1 - config.Epsilon())) %
                        (ratio <= (1 + config.Epsilon())) % advantages;
    arma::mat dSurroLoss = -(dL1 + dL2);

    // Calculates the gradient for Normal Distribution (mu and sigma)
    arma::mat dmu = (observation.t() - mu) / (arma::square(sigma)) % ratio;
    arma::mat dsigma = -1.0 / sigma +
                       arma::square(observation.t() - mu) / arma::pow(sigma, 3)
                       % ratio;

    // Calculates the gradient for activations (tanh and softplus)
    arma::mat dTanh, dSoftP;
    // in Tanh activation, the activation is provided as input in Deriv
    ann::TanhFunction::Deriv(mu, dTanh);
    // in Softplus activation, the input data is provided as input in Deriv
    ann::SoftplusFunction::Deriv(actionParameter.row(1), dSoftP);

    // calculation of final gradient using chain rule
    arma::mat grad1, grad2;
    grad1 = dTanh % dmu % dSurroLoss;
    grad2 = dSoftP % dsigma % dSurroLoss;
    arma::mat grad = arma::join_cols(grad1, grad2);

    // since empty loss is used, we give the gradient as input to Backward(),
    // instead of target.
    actorNetwork.Backward(sampledStates, grad, actorGradients);

    #if ENS_VERSION_MAJOR == 1
    actorUpdater.Update(actorNetwork.Parameters(), config.StepSize(),
                        actorGradients);
    #else
    actorUpdatePolicy->Update(actorNetwork.Parameters(), config.StepSize(),
                               actorGradients);
    #endif
  }

  // Update the oldActorNetwork, synchronize the parameter.
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
  arma::mat actionParameter, sigma, mu;

//  std::cout << "state: " << state.Encode() << std::endl;

  actorNetwork.Predict(state.Encode(), actionParameter);

  ann::TanhFunction::Fn(actionParameter.row(0), mu);
  ann::SoftplusFunction::Fn(actionParameter.row(1), sigma);

//  std::cout << "mu sigma: "<< mu << " " << sigma << std::endl;

  ann::NormalDistribution normalDist
      = ann::NormalDistribution(mu, sigma);

  ActionType action;
  action.action[0] = normalDist.Sample()[0];

//  std::cout << "action: " << action.action << std::endl;

  // Interact with the environment to advance to next state.
  StateType nextState;
  double reward = environment.Sample(state, action, nextState);

//  std::cout << "reward: " << reward << std::endl;

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
  while (!environment.IsTerminal(state))
  {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;

    totalReturn += Step();

    steps++;
    totalSteps++;

    if (deterministic)
      continue;

    if (steps > 0 && totalSteps % config.UpdateInterval() == 0)
    {
      Update();
      replayMethod.Clear();
    }
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack
#endif
