/**
 * @file methods/reinforcement_learning/eql_impl.hpp
 * @author Nanubala Gnana Sai
 *
 * This file is the implementation of Envelope Q-Learning class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_EQL_IMPL_HPP
#define MLPACK_METHODS_RL_EQL_IMPL_HPP

#include "eql.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::EQL(TrainingConfig& config,
       NetworkType& network,
       PolicyType& policy,
       ReplayType& replayMethod,
       UpdaterType updater,
       EnvironmentType environment):
    config(config),
    learningNetwork(network),
    policy(policy),
    replayMethod(replayMethod),
    updater(std::move(updater)),
    #if ENS_VERSION_MAJOR >= 2
    updatePolicy(NULL),
    #endif
    environment(std::move(environment)),
    totalSteps(0),
    deterministic(false)
{
  // To copy over the network structure.
  targetNetwork = learningNetwork;

  // Set up q-learning network.
  if (learningNetwork.Parameters().is_empty())
    learningNetwork.ResetParameters();

  targetNetwork.ResetParameters();

  #if ENS_VERSION_MAJOR == 1
  this->updater.Initialize(learningNetwork.Parameters().n_rows,
                           learningNetwork.Parameters().n_cols);
  #else
  this->updatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->updater,
                                   learningNetwork.Parameters().n_rows,
                                   learningNetwork.Parameters().n_cols);
  #endif

  // Initialize the target network with the parameters of learning network.
  targetNetwork.Parameters() = learningNetwork.Parameters();
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::~EQL()
{
  #if ENS_VERSION_MAJOR >= 2
  delete updatePolicy;
  #endif
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
arma::uvec EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::BestAction(const arma::mat& actionValues)
{
  // Take best possible action at a particular instance.
  arma::uvec bestActions(actionValues.n_cols);
  arma::rowvec maxActionValues = arma::max(actionValues, 0);
  for (size_t i = 0; i < actionValues.n_cols; ++i)
  {
    bestActions(i) = arma::as_scalar(
        arma::find(actionValues.col(i) == maxActionValues[i], 1));
  }
  return bestActions;
};

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
void EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType,
  ReplayType
>::TrainAgent()
{
  // Start experience replay.

  // Sample from previous experience.
  arma::mat sampledStates;
  std::vector<ActionType> sampledActions;
  arma::mat sampledRewardLists;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewardLists,
      sampledNextStates, isTerminal);

  size_t batchSize = sampledStates.n_cols;
  size_t extendedSize = numWeights * batchSize;
  size_t actionSize = sampledActions.n_rows;

  // Generate a repository of preference vectors.
  // TODO: The distribution should be set by user.
  arma::mat weightSpace = arma::randn(EnvironmentType::rewardSize, numWeights);
  weightSpace = arma::normalise(arma::abs(weightSpace), 1, 1);

  // Extended weight space. Each weight vector is repeated
  // batchSize number of times.
  arma::mat extWeightSpace = [&]()
  {
    arma::mat retval(EnvironmentType::rewardSize, extendedSize);
    size_t colIdx = 0;
    size_t start = 0;

    while (colIdx < numWeights)
    {
      retval.submat(arma::span(0, EnvironmentType::rewardSize),
                    arma::span(start, start + batchSize - 1)) =
          arma::repmat(weightSpace.col(colIdx), 1, batchSize);
      start += batchSize;
      ++colIdx;
    }

    return retval;
  }();

  // Every combination of weights and states. (rewardSize + stateSize, extendedSize).
  arma::mat sampledInputs =
      arma::join_cols(arma::repmat(sampledStates, 1, batchSize), extWeightSpace);

  arma::mat nextActionValues(actionSize * rewardSize, extendedSize);
  config.DoubleQLearning() ? learningEQLNetwork.Predict(nextStatePreference, nextActionValues)
                           : targetEQLNetwork.Predict(nextStatePreference, nextActionValues);

  arma::cube nextActionValues(nextActionValuesTmp.memptr(), actionSize, rewardSize, extendedSize);
  arma::uvec bestActions = BestAction(nextActionValues);

  arma::mat target;
  learningNetwork.Forward(sampledStates, target);

  double discount = std::pow(config.Discount(), replayMethod.NSteps());

  for (size_t i = 0; i < sampledNextStates.n_cols; ++i)
  {
    target(sampledActions[i].action, i) = sampledRewards(i) + discount *
        nextActionValues(bestActions(i), i) * (1 - isTerminal[i]);
  }

  // Learn from experience.
  arma::mat gradients;
  learningNetwork.Backward(sampledStates, target, gradients);

  replayMethod.Update(target, sampledActions, nextActionValues, gradients);

  #if ENS_VERSION_MAJOR == 1
  updater.Update(learningNetwork.Parameters(), config.StepSize(), gradients);
  #else
  updatePolicy->Update(learningNetwork.Parameters(), config.StepSize(),
      gradients);
  #endif

  if (config.NoisyEQL() == true)
  {
    learningNetwork.ResetNoise();
    targetNetwork.ResetNoise();
  }
  // Update target network.
  if (totalSteps % config.TargetNetworkSyncInterval() == 0)
    targetNetwork.Parameters() = learningNetwork.Parameters();

  if (totalSteps > config.ExplorationSteps())
    policy.Anneal();
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
void EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType,
  ReplayType
>::SelectAction()
{
  // Stores the Q vector of each action.
  arma::mat actionMatrix;
  // Get the unrolled form of the matrix.
  learningNetwork.Predict(state.Encode(), actionMatrix);
  actionMatrix.resize(ActionType::size, EnvironmentType::rewardSize);

  arma::vec utilityValue = actionMatrix * preference;
  // Select an action according to the behavior policy.
  action = policy.Sample(utilityValue, deterministic, config.NoisyEQL());
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType,
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

    TrainAgent();
  }
  return totalReturn;
}

} // namespace rl
} // namespace mlpack

#endif
