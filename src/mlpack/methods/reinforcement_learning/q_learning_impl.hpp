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
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::QLearning(TrainingConfig config,
             NetworkType network,
             PolicyType policy,
             ReplayType replayMethod,
             UpdaterType updater,
             EnvironmentType environment):
    config(std::move(config)),
    learningNetwork(std::move(network)),
    updater(std::move(updater)),
    #if ENS_VERSION_MAJOR >= 2
    updatePolicy(NULL),
    #endif
    policy(std::move(policy)),
    replayMethod(std::move(replayMethod)),
    environment(std::move(environment)),
    totalSteps(0),
    deterministic(false)
{
  // Set up q-learning network.
  if (learningNetwork.Parameters().is_empty())
    learningNetwork.ResetParameters();

  #if ENS_VERSION_MAJOR == 1
  this->updater.Initialize(learningNetwork.Parameters().n_rows,
                           learningNetwork.Parameters().n_cols);
  #else
  this->updatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->updater,
                                   learningNetwork.Parameters().n_rows,
                                   learningNetwork.Parameters().n_cols);
  #endif

  targetNetwork = learningNetwork;
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::~QLearning()
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
arma::Col<size_t> QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::BestAction(const arma::mat& actionValues)
{
  // Take best possible action at a particular instance.
  arma::Col<size_t> bestActions(actionValues.n_cols);
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
double QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType,
  ReplayType
>::Step()
{
  // Get the action value for each action at current state.
  arma::colvec actionValue;
  learningNetwork.Predict(state.Encode(), actionValue);

  // Select an action according to the behavior policy.
  ActionType action = policy.Sample(actionValue, deterministic);

  // Interact with the environment to advance to next state.
  StateType nextState;
  double reward = environment.Sample(state, action, nextState);

  // Store the transition for replay.
  replayMethod.Store(state, action, reward,
      nextState, environment.IsTerminal(nextState));

  // Update current state.
  state = nextState;

  if (deterministic || totalSteps < config.ExplorationSteps())
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
  arma::mat nextActionValues;
  targetNetwork.Predict(sampledNextStates, nextActionValues);

  arma::Col<size_t> bestActions;
  if (config.DoubleQLearning())
  {
    // If use double Q-Learning, use learning network to select the best action.
    arma::mat nextActionValues;
    learningNetwork.Predict(sampledNextStates, nextActionValues);
    bestActions = BestAction(nextActionValues);
  }
  else
  {
    bestActions = BestAction(nextActionValues);
  }

  // Compute the update target.
  arma::mat target;
  learningNetwork.Forward(sampledStates, target);
  /**
   * If the agent is at a terminal state, then we don't need to add the
   * discounted reward. At terminal state, the agent wont perform any
   * action.
   */
  for (size_t i = 0; i < sampledNextStates.n_cols; ++i)
  {
    if (isTerminal[i])
    {
      target(sampledActions(i), i) = sampledRewards(i);
    }
    else
    {
      target(sampledActions(i), i) = sampledRewards(i) + config.Discount() *
          nextActionValues(bestActions(i), i);
    }
  }

  // Learn from experience.
  arma::mat gradients;
  learningNetwork.Backward(target, gradients);

  replayMethod.Update(target, sampledActions, nextActionValues, gradients);

  #if ENS_VERSION_MAJOR == 1
  updater.Update(learningNetwork.Parameters(), config.StepSize(), gradients);
  #else
  updatePolicy->Update(learningNetwork.Parameters(), config.StepSize(),
      gradients);
  #endif

  return reward;
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
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

  // Running until get to the terminal state.
  while (!environment.IsTerminal(state))
  {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;

    totalReturn += Step();
    steps++;

    if (deterministic)
      continue;

    totalSteps++;

    // Update target network
    if (totalSteps % config.TargetNetworkSyncInterval() == 0)
      targetNetwork = learningNetwork;

    if (totalSteps > config.ExplorationSteps())
      policy.Anneal();
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack

#endif

