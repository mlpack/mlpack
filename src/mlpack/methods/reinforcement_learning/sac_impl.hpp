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
  typename PolicyType,
  typename ReplayType
>
SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::SAC(TrainingConfig& config,
       QNetworkType& learningQ1Network,
       QNetworkType& learningQ2Network,
       PolicyNetworkType& policyNetwork,
       PolicyType& policy,
       ReplayType& replayMethod,
       UpdaterType qNetworkUpdater,
       UpdaterType policyNetworkUpdater,
       EnvironmentType environment):
  config(config),
  learningQ1Network(learningQ1Network),
  learningQ2Network(learningQ2Network),
  policyNetwork(policyNetwork),
  policy(policy),
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
  if (learningQ2Network.Parameters().is_empty())
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
  typename PolicyType,
  typename ReplayType
>
SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  PolicyType,
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
  typename PolicyType,
  typename ReplayType
>
void SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Update()
{
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Step()
{
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double SAC<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
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
