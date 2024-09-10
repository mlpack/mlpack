/**
 * @file methods/reinforcement_learning/ddpg_impl.hpp
 * @author Tarek Elsayed
 *
 * This file is the implementation of DDPG class, which implements the
 * Deep Deterministic Policy Gradient algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_DDPG_IMPL_HPP
#define MLPACK_METHODS_RL_DDPG_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "ddpg.hpp"

namespace mlpack {

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename NoiseType,
  typename UpdaterType,
  typename ReplayType
>
DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  NoiseType,
  UpdaterType,
  ReplayType
>::DDPG(TrainingConfig& config,
       QNetworkType& learningQNetwork,
       PolicyNetworkType& policyNetwork,
       NoiseType& noise,
       ReplayType& replayMethod,
       UpdaterType qNetworkUpdater,
       UpdaterType policyNetworkUpdater,
       EnvironmentType environment):
  config(config),
  learningQNetwork(learningQNetwork),
  policyNetwork(policyNetwork),
  noise(noise),
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
  // Reset the noise instance.
  noise.reset();

  // Set up q-learning and policy networks.
  targetPNetwork = policyNetwork;
  targetQNetwork = learningQNetwork;

  // Reset all the networks.
  // Note: the q and policy networks have an if condition before reset.
  // This is because we don't want to reset a loaded(possibly pretrained) model
  // passed using this constructor.
  const size_t envSampleSize = environment.InitialSample().Encode().n_elem;
  if (policyNetwork.Parameters().n_elem != envSampleSize)
    policyNetwork.Reset(envSampleSize);

  targetPNetwork.Reset(envSampleSize);

  const size_t networkSize = envSampleSize +
      policyNetwork.Network()[policyNetwork.Network().size() - 1]->OutputSize();

  if (learningQNetwork.Parameters().n_elem != networkSize)
    learningQNetwork.Reset(networkSize);

  targetQNetwork.Reset(networkSize);

  #if ENS_VERSION_MAJOR == 1
  this->qNetworkUpdater.Initialize(learningQNetwork.Parameters().n_rows,
                                   learningQNetwork.Parameters().n_cols);
  #else
  this->qNetworkUpdatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->qNetworkUpdater,
                                   learningQNetwork.Parameters().n_rows,
                                   learningQNetwork.Parameters().n_cols);
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

  // Copy over the learning networks to their respective target networks.
  targetQNetwork.Parameters() = learningQNetwork.Parameters();
  targetPNetwork.Parameters() = policyNetwork.Parameters();
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename NoiseType,
  typename UpdaterType,
  typename ReplayType
>
DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  NoiseType,
  UpdaterType,
  ReplayType
>::~DDPG()
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
  typename NoiseType,
  typename UpdaterType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  NoiseType,
  UpdaterType,
  ReplayType
>::SoftUpdate(double rho)
{
  targetQNetwork.Parameters() = (1 - rho) * targetQNetwork.Parameters() +
      rho * learningQNetwork.Parameters();
  targetPNetwork.Parameters() = (1 - rho) * targetPNetwork.Parameters() +
      rho * policyNetwork.Parameters();
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename NoiseType,
  typename UpdaterType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  NoiseType,
  UpdaterType,
  ReplayType
>::Update()
{
  // Sample from previous experience.
  arma::mat sampledStates;
  std::vector<ActionType> sampledActions;
  arma::rowvec sampledRewards;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
      sampledNextStates, isTerminal);

  // Critic network update.

  // Use the target actor to obtain the next actions.
  arma::mat nextStateActions;
  targetPNetwork.Predict(sampledNextStates, nextStateActions);

  arma::mat targetQInput = arma::join_vert(nextStateActions,
      sampledNextStates);
  arma::rowvec Q;
  targetQNetwork.Predict(targetQInput, Q);
  arma::rowvec nextQ = sampledRewards + config.Discount()
      * ((1 - isTerminal) % Q);

  arma::mat sampledActionValues(action.size, sampledActions.size());
  for (size_t i = 0; i < sampledActions.size(); i++)
    sampledActionValues.col(i) = ConvTo<arma::colvec>::From
                                 (sampledActions[i].action);
  arma::mat learningQInput = arma::join_vert(sampledActionValues,
      sampledStates);
  learningQNetwork.Forward(learningQInput, Q);

  arma::mat gradQLoss;
  lossFunction.Backward(Q, nextQ, gradQLoss);

  // Update the critic network.
  arma::mat gradientQ;
  learningQNetwork.Backward(learningQInput, gradQLoss, gradientQ);
  #if ENS_VERSION_MAJOR == 1
  qNetworkUpdater.Update(learningQNetwork.Parameters(), config.StepSize(),
      gradientQ);
  #else
  qNetworkUpdatePolicy->Update(learningQNetwork.Parameters(),
      config.StepSize(), gradientQ);
  #endif

  // Actor network update.

  // Get the size of the first hidden layer in the Q network.
  size_t hidden1 = learningQNetwork.Network()[0]->OutputSize();

  arma::mat gradient;
  for (size_t i = 0; i < sampledStates.n_cols; i++)
  {
    arma::mat grad, gradQ, q;
    arma::colvec singleState = sampledStates.col(i);
    arma::colvec singlePi;
    policyNetwork.Forward(singleState, singlePi);
    arma::colvec input = arma::join_vert(singlePi, singleState);
    arma::mat weightLastLayer;

    // Note that we can use an empty matrix for the backwards pass, since the
    // networks use EmptyLoss.
    learningQNetwork.Forward(input, q);
    learningQNetwork.Backward(input, arma::mat("-1"), gradQ);
    weightLastLayer = arma::reshape(learningQNetwork.Parameters().
          rows(0, hidden1 * singlePi.n_rows - 1), hidden1, singlePi.n_rows);

    arma::colvec gradQBias = gradQ(input.n_rows * hidden1, 0,
        arma::size(hidden1, 1));
    arma::mat gradPolicy = weightLastLayer.t() * gradQBias;
    policyNetwork.Backward(singleState, gradPolicy, grad);
    if (i == 0)
    {
      gradient.copy_size(grad);
      gradient.fill(0.0);
    }
    gradient += grad;
  }
  gradient /= sampledStates.n_cols;

  #if ENS_VERSION_MAJOR == 1
  policyUpdater.Update(policyNetwork.Parameters(), config.StepSize(), gradient);
  #else
  policyNetworkUpdatePolicy->Update(policyNetwork.Parameters(),
      config.StepSize(), gradient);
  #endif

  // Update target networks
  if (totalSteps % config.TargetNetworkSyncInterval() == 0)
    SoftUpdate(config.Rho());
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename NoiseType,
  typename UpdaterType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  NoiseType,
  UpdaterType,
  ReplayType
>::SelectAction()
{
  // Get the action at current state, from policy.
  arma::colvec outputAction;
  policyNetwork.Predict(state.Encode(), outputAction);

  if (!deterministic)
  {
    arma::colvec sample = noise.sample() * 0.1;
    sample = arma::clamp(sample, -0.25, 0.25);
    outputAction = outputAction + sample;
  }
  action.action = ConvTo<std::vector<double>>::From(outputAction);
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename NoiseType,
  typename UpdaterType,
  typename ReplayType
>
double DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  NoiseType,
  UpdaterType,
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
    SelectAction();

    // Interact with the environment to advance to next state.
    StateType nextState;
    double reward = environment.Sample(state, action, nextState);

    totalReturn += reward;
    steps++;
    totalSteps++;

    // Store the transition for replay.
    replayMethod.Store(state, action, reward, nextState,
        environment.IsTerminal(nextState), config.Discount());

    // Update current state.
    state = nextState;

    if (deterministic || totalSteps < config.ExplorationSteps())
      continue;
    for (size_t i = 0; i < config.UpdateInterval(); i++)
      Update();
  }
  return totalReturn;
}

} // namespace mlpack
#endif
