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
  typename UpdaterType
>
PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType
>::PPO(TrainingConfig& config,
       ActorNetworkType& actor,
       CriticNetworkType& critic,
       UpdaterType updater,
       EnvironmentType environment):
  config(config),
  actorNetwork(actor),
  criticNetwork(critic),
  actorUpdater(std::move(updater)),
  #if ENS_VERSION_MAJOR >= 2
  actorUpdatePolicy(NULL),
  #endif
  criticUpdater(std::move(updater)),
  #if ENS_VERSION_MAJOR >= 2
  criticUpdatePolicy(NULL),
  #endif
  environment(std::move(environment)),
  totalSteps(0),
  deterministic(false)
{

  oldActorNetwork = actorNetwork;
  // Reset all the networks.
  // Note: the actor and critic networks have an if condition before reset.
  // This is because we don't want to reset a loaded(possibly pretrained) model
  // passed using this constructor.
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

  // Copy over the actor parameters to its older self.
  oldActorNetwork.Parameters() = actorNetwork.Parameters();
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType
>
PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType
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
  typename UpdaterType
>
void PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType
>::SelectAction()
{
  // Get action logits for each action at current state.
  arma::colvec actionLogit;
  actorNetwork.Predict(state.Encode(), actionLogit);

  arma::colvec prob;
  ann::Softmax<> softmax;
  softmax.Forward(actionLogit, prob);

  // Calculating cumulative probablity.
  for (size_t i = 1; i<prob.n_rows; i++)
    prob[i] += prob[i-1];
  
  // Sampling action from cumulative probablity.
  double randValue = arma::randu<double>();
  for (size_t i = 0; i<prob.n_rows; i++)
    if (randValue <= prob[i])
    {
      action.action = static_cast<decltype(action.action)>(i);
      return;
    }  
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType
>
void PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType
>::Update()
{
  arma::mat sampledStates = state.Encode();
  std::vector<ActionType> sampledActions = {action};
  arma::rowvec sampledRewards = {reward};
  arma::irowvec isTerminal = {done};
  arma::mat criticGradients, actorGradients;
  
  arma::rowvec actionValues, advantages;
  criticNetwork.Forward(sampledStates, actionValues);

  arma::rowvec discountedRewards(sampledRewards.n_cols);
  arma::rowvec nextActionValue;
  criticNetwork.Predict(nextState.Encode(), nextActionValue);
  double R = nextActionValue(0);
  for (size_t i = sampledRewards.n_cols; i > 0; --i)
  {
    if (isTerminal[i-1])
      R = 0;
    R = sampledRewards[i-1] + R * config.Discount();
    discountedRewards[sampledRewards.n_cols - i] = R;
  }

  advantages = arma::conv_to<arma::mat>::
               from(discountedRewards) - actionValues;
  
  // since empty loss is used, we give the gradient as input to Backward(),
  // instead of target.
  arma::mat dLossCritic = - (advantages);
  criticNetwork.Backward(sampledStates, dLossCritic, criticGradients);
  #if ENS_VERSION_MAJOR == 1
  criticUpdater.Update(criticNetwork.Parameters(), config.StepSize(),
                       criticGradients);
  #else
  criticUpdatePolicy->Update(criticNetwork.Parameters(), config.StepSize(),
                             criticGradients);
  #endif

  ann::Softmax<> softmax;
  arma::mat actionLogit, actionProb;

  oldActorNetwork.Forward(sampledStates, actionLogit);
  softmax.Forward(actionLogit, actionProb);
  arma::mat oldProb = actionProb.row(action.action);

  actorNetwork.Forward(sampledStates, actionLogit);
  softmax.Forward(actionLogit, actionProb);
  arma::mat prob = actionProb.row(action.action);

  arma::mat ratio = prob / oldProb;

  arma::mat L1 = ratio % advantages;
  arma::mat L2 = arma::clamp(ratio, 1 - config.Epsilon(),
                            1 + config.Epsilon()) % advantages;
  // arma::mat surroLoss = -arma::min(L1, L2);

  // Calculates the gradient for Surrogate Loss
  arma::mat dL1 = (L1 < L2) % (advantages / oldProb);
  arma::mat dL2 = (L1 >= L2) % (ratio >= (1 - config.Epsilon())) %
                      (ratio <= (1 + config.Epsilon())) % (advantages/oldProb);
  arma::mat dSurroLoss = -(dL1 + dL2);

  arma::mat dLoss(action.size, sampledActions.size(), arma::fill::zeros);
  for (size_t i = 0; i < sampledActions.size(); i++)
    dLoss(sampledActions[i].action, i) = dSurroLoss(0, i);
  
  arma::mat dGrad;
  softmax.Backward(actionProb, dLoss, dGrad);

  // since empty loss is used, we give the gradient as input to Backward(),
  // instead of target.
  actorNetwork.Backward(sampledStates, dGrad, actorGradients);

  #if ENS_VERSION_MAJOR == 1
  actorUpdater.Update(actorNetwork.Parameters(), config.StepSize(),
                      actorGradients);
  #else
  actorUpdatePolicy->Update(actorNetwork.Parameters(), config.StepSize(),
                              actorGradients);
  #endif

  // Update the oldActorNetwork, synchronize the parameter.
  oldActorNetwork.Parameters() = actorNetwork.Parameters();
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType
>
double PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType
>::Episode()
{
  // Get the initial state from environment.
  state = environment.InitialSample();

  // Track the return of this episode.
  double totalReturn = 0.0;

  // Running until get to the terminal state.
  while (!environment.IsTerminal(state))
  {
    SelectAction();

    // Interact with the environment to advance to next state.
    reward = environment.Sample(state, action, nextState);

    totalReturn += reward;
    totalSteps++;
    done = environment.IsTerminal(nextState);

    if (deterministic)
      continue;

    Update();

    // Update current state.
    state = nextState;
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack
#endif
