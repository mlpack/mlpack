/**
 * @file sac_impl.hpp
 * @author Nishant Kumar
 *
 * This file is the implementation of SAC class, which implements the
 * soft actor-critic algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_SAC_IMPL_HPP
#define MLPACK_METHODS_RL_SAC_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "sac.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::SAC(TrainingConfig& config,
       QNetworkType& learningQ1Network,
       PolicyNetworkType& policyNetwork,
       ReplayType& replayMethod,
       UpdaterType qNetworkUpdater,
       UpdaterType policyNetworkUpdater,
       EnvironmentType environment):
  config(config),
  learningQ1Network(learningQ1Network),
  policyNetwork(policyNetwork),
  replayMethod(replayMethod),
  qNetworkUpdater(std::move(qNetworkUpdater)),
  #if ENS_VERSION_MAJOR >= 2
  qNetworkUpdatePolicy(NULL),
  #endif
  policyNetworkUpdater(std::move(policyNetworkUpdater)),
  #if ENS_VERSION_MAJOR >= 2
  policyNetworkUpdatePolicy(NULL),
  #endif
  environment(std::move(environment)),
  totalSteps(0),
  deterministic(false)
{
  // Set up q-learning and policy networks.
  if (learningQ1Network.Parameters().is_empty())
    learningQ1Network.ResetParameters();
  learningQ2Network = learningQ1Network;
  learningQ2Network.ResetParameters();
  if (policyNetwork.Parameters().is_empty())
    policyNetwork.ResetParameters();

  #if ENS_VERSION_MAJOR == 1
  this->qNetworkUpdater.Initialize(learningQ1Network.Parameters().n_rows,
                                   learningQ1Network.Parameters().n_cols);
  #else
  this->qNetworkUpdatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->qNetworkUpdater,
                                   learningQ1Network.Parameters().n_rows,
                                   learningQ1Network.Parameters().n_cols);
  #endif

  #if ENS_VERSION_MAJOR == 1
  this->policyNetworkUpdater.Initialize(policyNetwork.Parameters().n_rows,
                                        policyNetwork.Parameters().n_cols);
  #else
  this->policyNetworkUpdatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->policyNetworkUpdater,
                                   policyNetwork.Parameters().n_rows,
                                   policyNetwork.Parameters().n_cols);
  #endif

  targetQ1Network = learningQ1Network;
  targetQ2Network = learningQ2Network;
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::~SAC()
{
  #if ENS_VERSION_MAJOR >= 2
  delete qNetworkUpdatePolicy;
  delete policyNetworkUpdatePolicy;
  #endif
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::SoftUpdate()
{
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::Update()
{
  // Update target network
  if (totalSteps % config.TargetNetworkSyncInterval() == 0)
    SoftUpdate();
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::SelectAction()
{
  // Get the action at current state, from policy.
  arma::colvec outputAction;
  policyNetwork.Predict(state.Encode(), outputAction);

  if (!deterministic)
  {
    arma::colvec noise = arma::randu<arma::colvec>(outputAction.n_rows) * 0.1;
    noise = arma::clamp(noise, -0.25, 0.25);
    outputAction = outputAction + noise;
  }
  action.action[0] = outputAction[0];
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
double SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
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
    StateType nextState;
    double reward = environment.Sample(state, action, nextState);

    totalReturn += reward;
    totalSteps++;

    // Store the transition for replay.
    replayMethod.Store(state, action, reward, nextState,
        environment.IsTerminal(nextState), config.Discount());

    // Update current state.
    state = nextState;

    if (deterministic || totalSteps < config.ExplorationSteps())
      continue;
    Update();
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack
#endif
