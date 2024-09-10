/**
 * @file methods/reinforcement_learning/td3_impl.hpp
 * @author Tarek Elsayed
 *
 * This file is the implementation of TD3 class, which implements the
 * Twin Delayed Deep Deterministic Policy Gradient algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_td3_IMPL_HPP
#define MLPACK_METHODS_RL_td3_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "td3.hpp"

namespace mlpack {

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
TD3<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::TD3(TrainingConfig& config,
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
  targetPNetwork = policyNetwork;
  targetQ1Network = learningQ1Network;
  learningQ2Network = learningQ1Network;
  targetQ2Network = learningQ2Network;

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

  if (learningQ1Network.Parameters().n_elem != networkSize)
  {
    learningQ1Network.Reset(networkSize);
    learningQ2Network.Reset(networkSize);
  }
  targetQ1Network.Reset(networkSize);
  targetQ2Network.Reset(networkSize);

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

  // Copy over the learning networks to their respective target networks.
  targetQ1Network.Parameters() = learningQ1Network.Parameters();
  targetQ2Network.Parameters() = learningQ2Network.Parameters();
  targetPNetwork.Parameters() = policyNetwork.Parameters();
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
TD3<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::~TD3()
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
void TD3<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::SoftUpdate(double rho)
{
  targetQ1Network.Parameters() = (1 - rho) * targetQ1Network.Parameters() +
      rho * learningQ1Network.Parameters();
  targetQ2Network.Parameters() = (1 - rho) * targetQ2Network.Parameters() +
      rho * learningQ2Network.Parameters();
  targetPNetwork.Parameters() = (1 - rho) * targetPNetwork.Parameters() +
      rho * policyNetwork.Parameters();
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void TD3<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
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

  // Compute the estimated next Q-values using the target Q-networks.
  arma::mat targetQInput = arma::join_vert(nextStateActions,
      sampledNextStates);
  arma::rowvec Q1, Q2;
  targetQ1Network.Predict(targetQInput, Q1);
  targetQ2Network.Predict(targetQInput, Q2);
  arma::rowvec nextQ = sampledRewards + config.Discount() * ((1 - isTerminal)
      % min(Q1, Q2));

  arma::mat sampledActionValues(action.size, sampledActions.size());
  for (size_t i = 0; i < sampledActions.size(); i++)
    sampledActionValues.col(i) = ConvTo<arma::colvec>::From
                                 (sampledActions[i].action);
  arma::mat learningQInput = arma::join_vert(sampledActionValues,
      sampledStates);
  learningQ1Network.Forward(learningQInput, Q1);
  learningQ2Network.Forward(learningQInput, Q2);

  arma::mat gradQ1Loss, gradQ2Loss;
  lossFunction.Backward(Q1, nextQ, gradQ1Loss);
  lossFunction.Backward(Q2, nextQ, gradQ2Loss);

  // Sum both losses
  arma::mat combinedLoss = gradQ1Loss + gradQ2Loss;

  // Update the critic networks.
  arma::mat gradientQ1, gradientQ2;
  learningQ1Network.Backward(learningQInput, combinedLoss, gradientQ1);
  learningQ2Network.Backward(learningQInput, combinedLoss, gradientQ2);
  #if ENS_VERSION_MAJOR == 1
  qNetworkUpdater.Update(learningQ1Network.Parameters(), config.StepSize(),
      gradientQ1);
  #else
  qNetworkUpdatePolicy->Update(learningQ1Network.Parameters(),
      config.StepSize(), gradientQ1);
  #endif
  #if ENS_VERSION_MAJOR == 1
  qNetworkUpdater.Update(learningQ2Network.Parameters(), config.StepSize(),
      gradientQ2);
  #else
  qNetworkUpdatePolicy->Update(learningQ2Network.Parameters(),
      config.StepSize(), gradientQ2);
  #endif

  // Actor network update.

  if (totalSteps % config.TargetNetworkSyncInterval() == 0)
  {
    // Get the size of the first hidden layer in the Q network.
    size_t hidden1 = learningQ1Network.Network()[0]->OutputSize();

    arma::mat gradient;
    for (size_t i = 0; i < sampledStates.n_cols; i++)
    {
        arma::mat grad, gradQ, q;
        arma::colvec singleState = sampledStates.col(i);
        arma::colvec singlePi;
        policyNetwork.Forward(singleState, singlePi);
        arma::colvec input = arma::join_vert(singlePi, singleState);
        arma::mat weightLastLayer;

        // Note that we can use an empty matrix for the backwards pass, since
        // the networks use EmptyLoss.
        learningQ1Network.Forward(input, q);
        learningQ1Network.Backward(input, arma::mat("-1"), gradQ);
        weightLastLayer = arma::reshape(learningQ1Network.Parameters().
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
    policyUpdater.Update(policyNetwork.Parameters(), config.StepSize(),
        gradient);
    #else
    policyNetworkUpdatePolicy->Update(policyNetwork.Parameters(),
        config.StepSize(), gradient);
    #endif

    // Update target networks
    SoftUpdate(config.Rho());
  }
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void TD3<
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
    arma::colvec noise;
    noise.randn(outputAction.n_rows) * 0.1;
    noise = arma::clamp(noise, -0.25, 0.25);
    outputAction = outputAction + noise;
  }
  action.action = ConvTo<std::vector<double>>::From(outputAction);
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
double TD3<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
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
