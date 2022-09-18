/**
 * @file ppo.hpp
 * @author Eshaan Agarwal
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
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/core/util/log.hpp>

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
  const size_t envSampleSize = environment.InitialSample().Encode().n_elem;
  std::cout<< "env sample size "<< envSampleSize<< std::endl;
  std::cout<< "criic network elements "<< criticNetwork.Parameters().n_elem << std::endl;
  // if (actorNetwork.Parameters().n_elem != envSampleSize)
  //   actorNetwork.Reset(envSampleSize);

  // if (criticNetwork.Parameters().n_elem != envSampleSize)
  //   criticNetwork.Reset(envSampleSize);

  std::cout<< "criic network dimensions "<< criticNetwork.Parameters().n_rows << " "<< criticNetwork.Parameters().n_cols << std::endl;
  std::cout<< "actor network dimensions "<< actorNetwork.Parameters().n_rows << " "<< actorNetwork.Parameters().n_cols << std::endl;

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

  arma::colvec cumulativeProb;
  ann::Softmax softmax;
  softmax.Forward(actionLogit, cumulativeProb);

  for (size_t i = 1; i < cumulativeProb.n_rows; i++)
      cumulativeProb[i] += cumulativeProb[i-1];

  // Sampling action from cumulative probablity.
  for (size_t actionIdx = 0; actionIdx < cumulativeProb.n_rows; ++actionIdx)
  {
    if (mlpack::math::Random() <= cumulativeProb[actionIdx])
    {
      action.action = static_cast<decltype(action.action)>(actionIdx);
      return;
    }
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

  std::cout<< "critic 1 old weights \n" << criticNetwork.Parameters() <<std::endl;

  arma::rowvec actionValues, advantages;
  criticNetwork.Forward(sampledStates, actionValues);

  std::cout << "critic network' action values \n"<<actionValues<<std::endl;

  arma::rowvec discountedRewards(sampledRewards.n_cols);
  arma::rowvec nextActionValue;
  criticNetwork.Predict(nextState.Encode(), nextActionValue);
  double R = nextActionValue(0);
  for (size_t i = sampledRewards.n_cols; i > 0; --i)
  {
    if (isTerminal[i - 1])
      R = 0;
    R = sampledRewards[i - 1] + R * config.Discount();
    discountedRewards[sampledRewards.n_cols - i] = R;
  }

  advantages = arma::conv_to<arma::mat>::
      from(discountedRewards) - actionValues;

  // since empty loss is used, we give the gradient as input to Backward(),
  // instead of target.
  // arma::mat dLossCritic = - (advantages);
  arma::mat dLossCritic = - (advantages) * actionValues;

  std::cout << "critic network' loss values \n"<<dLossCritic<<std::endl;
  

  criticNetwork.Backward(sampledStates, dLossCritic, criticGradients);

  std::cout<< "critic gradients \n" << criticGradients<<std::endl;

  std::cout<< "critic old weights \n" << criticNetwork.Parameters() <<std::endl;


  #if ENS_VERSION_MAJOR == 1
  criticUpdater.Update(criticNetwork.Parameters(), config.StepSize(),
                       criticGradients);
  #else
  criticUpdatePolicy->Update(criticNetwork.Parameters(), config.StepSize(),
                             criticGradients);
  #endif

  std::cout<< "critic updated weights \n" << criticNetwork.Parameters() <<std::endl;

  ann::LogSoftMax softmax;
  arma::mat actionLogit, actionProb;

  oldActorNetwork.Forward(sampledStates, actionLogit);
  softmax.Forward(actionLogit, actionProb);
  std::cout<<" actor forward actionlogit" << actionLogit <<std::endl;
  std::cout<<" old actor forward actionprobs" << actionProb <<std::endl;
  arma::mat oldProb = actionProb.row(action.action);

  actorNetwork.Forward(sampledStates, actionLogit);
  softmax.Forward(actionLogit, actionProb);

  std::cout<<" actor forward actionprob" << actionProb <<std::endl;
  arma::mat prob = actionProb.row(action.action);

  arma::mat ratio = prob / oldProb;

  arma::mat L1 = ratio % advantages;
  arma::mat L2 = arma::clamp(ratio, 1 - config.Epsilon(),
      1 + config.Epsilon()) % advantages;

  // Calculates the gradient for Surrogate Loss
  arma::mat dL1 = (L1 < L2) % (advantages / oldProb);
  arma::mat dL2 = (L1 >= L2) % (ratio >= (1 - config.Epsilon())) %
      (ratio <= (1 + config.Epsilon())) % (advantages/oldProb);
  arma::mat dSurroLoss = -(dL1 + dL2);

  arma::mat dLoss(action.size, sampledActions.size(), arma::fill::zeros);
  for (size_t i = 0; i < sampledActions.size(); i++)
    dLoss(sampledActions[i].action, i) = dSurroLoss(0, i);

  arma::mat dGrad;

  std::cout<<" actor loss" << dLoss <<std::endl;
  softmax.Backward(actionProb, dLoss, dGrad);

  // Since empty loss is used, we give the gradient as input to Backward(),
  // instead of target.

  std::cout<<" actor backward dGrad" << dGrad <<std::endl;
  actorNetwork.Backward(sampledStates, dGrad, actorGradients);

  std::cout<<" actor backward gradients" << actorGradients <<std::endl;

  arma::colvec oldweights=actorNetwork.Parameters();

  std::cout<< "actor weigh \n" << actorNetwork.Parameters() <<std::endl;


  #if ENS_VERSION_MAJOR == 1
  actorUpdater.Update(actorNetwork.Parameters(), config.StepSize(),
                      actorGradients);
  #else
  actorUpdatePolicy->Update(actorNetwork.Parameters(), config.StepSize(),
                            actorGradients);
  #endif

  arma::colvec updatedweights=actorNetwork.Parameters();
  std::cout<< "actor weight difference \n" << sum(updatedweights-oldweights) <<std::endl;


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

  std::cout<<"Initial State "<< state.Position() << " " << state.Velocity() << " " << state.Angle() <<" " << state.AngularVelocity() <<  std::endl;

  // Track the return of this episode.
  double totalReturn = 0.0;

  int i=0;
  // Running until get to the terminal state.
  while (!environment.IsTerminal(state))
  {
    SelectAction();

    // std::cout<<"Action selected "<<action.action<<std::endl;

    // Interact with the environment to advance to next state.
    reward = environment.Sample(state, action, nextState);

    // std::cout<<"Reward on Action "<<reward<<std::endl;

    totalReturn += reward;
    totalSteps++;
    done = environment.IsTerminal(nextState);

    if (deterministic)
      continue;

    Update();

    // Update current state.
    state = nextState;
    i++;

    if(i>1) break;
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack
#endif