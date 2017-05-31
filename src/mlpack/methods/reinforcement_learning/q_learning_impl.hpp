/**
 * @file q_learning_impl.hpp
 * @author Shangtong Zhang
 *
 * This file is the implementation of QLearning class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_Q_LEARNING_IMPL_HPP
#define MLPACK_METHODS_RL_Q_LEARNING_IMPL_HPP

#include "q_learning.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename NetworkType,
  typename OptimizerType,
  typename PolicyType,
  typename ReplayType
>
QLearning<
  EnvironmentType,
  NetworkType,
  OptimizerType,
  PolicyType,
  ReplayType
>::QLearning(NetworkType& network,
             OptimizerType& optimizer,
             double discount,
             PolicyType policy,
             ReplayType replayMethod,
             size_t targetNetworkSyncInterval,
             size_t explorationsSteps,
             bool doubleQLearning,
             size_t stepLimit,
             EnvironmentType environment):
    learningNetwork(network),
    optimizer(optimizer),
    discount(discount),
    policy(std::move(policy)),
    replayMethod(std::move(replayMethod)),
    targetNetworkSyncInterval(targetNetworkSyncInterval),
    explorationSteps(explorationsSteps),
    doubleQLearning(doubleQLearning),
    stepLimit(stepLimit),
    environment(std::move(environment)),
    totalSteps(0),
    deterministic(false)
{
  learningNetwork.ResetParameters();
  targetNetwork = learningNetwork;
}


template <
  typename EnvironmentType,
  typename NetworkType,
  typename OptimizerType,
  typename PolicyType,
  typename ReplayType
>
arma::icolvec QLearning<
  EnvironmentType,
  NetworkType,
  OptimizerType,
  PolicyType,
  ReplayType
>::BestAction(const arma::mat& actionValues)
{
  arma::icolvec bestActions(actionValues.n_cols);
  arma::rowvec maxActionValues = arma::max(actionValues, 0);
  for (size_t i = 0; i < actionValues.n_cols; ++i)
    bestActions(i) = arma::as_scalar(
        arma::find(actionValues.col(i) == maxActionValues[i], 1));
  return bestActions;
};

template <
  typename EnvironmentType,
  typename NetworkType,
  typename OptimizerType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double QLearning<
  EnvironmentType,
  NetworkType,
  OptimizerType,
  BehaviorPolicyType,
  ReplayType
>::Step()
{
  // Get the action value for each action at current state.
  arma::colvec actionValue = learningNetwork.Predict(state.Encode());

  // Select an action according to the behavior policy.
  ActionType action = policy.Sample(actionValue, deterministic);

  // Interact with the environment to advance to next state.
  StateType nextState;
  double reward = environment.Sample(state, action, nextState);

  // Store the transition for replay.
  replayMethod.Store(state, action, reward, nextState, environment.IsTerminal(nextState));

  // Update current state.
  state = nextState;

  if (deterministic || totalSteps < explorationSteps)
    return reward;

  // Start experience replay.

  // Sample from previous experience.
  arma::mat sampledStates;
  arma::icolvec sampledActions;
  arma::colvec sampledRewards;
  arma::mat sampledNextStates;
  arma::icolvec isTerminal;
  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
                      sampledNextStates, isTerminal);

  // Compute action value for next state with target network.
  arma::mat nextActionValues = targetNetwork.Predict(sampledNextStates);

  arma::icolvec bestActions;
  if (doubleQLearning)
    // If use double Q-Learning, use learning network to select the best action.
    bestActions = BestAction(learningNetwork.Predict(sampledNextStates));
  else
    bestActions = BestAction(nextActionValues);

  // Compute the update target.
  arma::mat target = learningNetwork.Predict(sampledStates);
  for (size_t i = 0; i < sampledNextStates.n_cols; ++i)
    target(sampledActions[i], i) = sampledRewards[i] +
        discount * (isTerminal[i] ? 0.0 : nextActionValues(bestActions[i], i));

  // Learn form experience.
  learningNetwork.Train(sampledStates, target, optimizer);

  return reward;
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename OptimizerType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double QLearning<
  EnvironmentType,
  NetworkType,
  OptimizerType,
  BehaviorPolicyType,
  ReplayType
>::Episode()
{
  // Get the initial state from environment.
  state = environment.InitialSample();

  // Track the steps in this episode.
  size_t steps = 0;

  // Track the return of this episode.
  double totalReturn = 0.0;

  // Running until get ot the terminal state.
  while (!environment.IsTerminal(state))
  {
    if (stepLimit && steps >= stepLimit)
      break;

    totalReturn += Step();
    steps++;

    if (deterministic)
      continue;

    totalSteps++;

    // Update target network
    if (totalSteps % targetNetworkSyncInterval == 0)
      targetNetwork = learningNetwork;

    if (totalSteps > explorationSteps)
      policy.Anneal();
  }
  return totalReturn;
}

} // namespace rl
} // namespace mlpack

#endif

