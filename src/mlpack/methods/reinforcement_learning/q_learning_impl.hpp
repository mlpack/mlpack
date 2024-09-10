/**
 * @file methods/reinforcement_learning/q_learning_impl.hpp
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
>::QLearning(TrainingConfig& config,
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
  if (learningNetwork.Parameters().n_elem !=
      environment.InitialSample().Encode().n_elem)
  {
    learningNetwork.Reset(environment.InitialSample().Encode().n_elem);
  }

  // Initialize the target network with the parameters of learning network.
  targetNetwork.Parameters() = learningNetwork.Parameters();
  targetNetwork.Reset(environment.InitialSample().Encode().n_elem);

  #if ENS_VERSION_MAJOR == 1
  this->updater.Initialize(learningNetwork.Parameters().n_rows,
                           learningNetwork.Parameters().n_cols);
  #else
  this->updatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->updater,
                                   learningNetwork.Parameters().n_rows,
                                   learningNetwork.Parameters().n_cols);
  #endif
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
  arma::rowvec maxActionValues = max(actionValues, 0);
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
void QLearning<
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
  arma::rowvec sampledRewards;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

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

  double discount = std::pow(config.Discount(), replayMethod.NSteps());

  /**
   * If the agent is at a terminal state, then we don't need to add the
   * discounted reward. At terminal state, the agent wont perform any
   * action.
   */
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

  if (config.NoisyQLearning() == true)
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
void QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType,
  ReplayType
>::TrainCategoricalAgent()
{
  // Start experience replay.

  // Sample from previous experience.
  arma::mat sampledStates;
  std::vector<ActionType> sampledActions;
  arma::rowvec sampledRewards;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
      sampledNextStates, isTerminal);

  size_t atomSize = config.AtomSize();
  arma::colvec support = arma::linspace<arma::colvec>(config.VMin(),
      config.VMax(), atomSize);

  size_t batchSize = sampledNextStates.n_cols;

  // Compute action value for next state with target network.
  arma::mat nextActionValues;
  targetNetwork.Predict(sampledNextStates, nextActionValues);

  arma::Col<size_t> nextAction;
  if (config.DoubleQLearning())
  {
    // If use double Q-Learning, use learning network to select the best action.
    arma::mat nextActionValues;
    learningNetwork.Predict(sampledNextStates, nextActionValues);
    nextAction = BestAction(nextActionValues);
  }
  else
  {
    nextAction = BestAction(nextActionValues);
  }

  arma::mat nextDists, nextDist(atomSize, batchSize);
  targetNetwork.Forward(sampledNextStates, nextDists);
  for (size_t i = 0; i < batchSize; ++i)
  {
    nextDist.col(i) = nextDists(nextAction(i) * atomSize, i,
        arma::size(atomSize, 1));
  }

  arma::mat tZ = (ConvTo<arma::mat>::From(config.Discount() *
      (support * (1 - isTerminal))).each_row() + sampledRewards);
  tZ = arma::clamp(tZ, config.VMin(), config.VMax());
  arma::mat b = (tZ - config.VMin()) / (config.VMax() - config.VMin()) *
      (atomSize - 1);
  arma::mat l = arma::floor(b);
  arma::mat u = arma::ceil(b);

  arma::mat projDistUpper = nextDist % (u - b);
  arma::mat projDistLower = nextDist % (b - l);

  arma::mat projDist = zeros<arma::mat>(arma::size(nextDist));
  for (size_t batchNo = 0; batchNo < batchSize; batchNo++)
  {
    for (size_t j = 0; j < atomSize; j++)
    {
      projDist(l(j, batchNo), batchNo) += projDistUpper(j, batchNo);
      projDist(u(j, batchNo), batchNo) += projDistLower(j, batchNo);
    }
  }
  arma::mat dists;
  learningNetwork.Forward(sampledStates, dists);
  arma::mat lossGradients = zeros<arma::mat>(arma::size(dists));
  for (size_t i = 0; i < batchSize; ++i)
  {
    lossGradients(sampledActions[i].action * atomSize, i,
        arma::size(atomSize, 1)) = -(projDist.col(i) / (1e-10 + dists(
        sampledActions[i].action * atomSize, i, arma::size(atomSize, 1))));
  }
  // Learn from experience.
  arma::mat gradients;
  learningNetwork.Backward(sampledStates, lossGradients, gradients);

  #if ENS_VERSION_MAJOR == 1
  updater.Update(learningNetwork.Parameters(), config.StepSize(), gradients);
  #else
  updatePolicy->Update(learningNetwork.Parameters(), config.StepSize(),
      gradients);
  #endif

  if (config.NoisyQLearning() == true)
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
void QLearning<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType,
  ReplayType
>::SelectAction()
{
  // Get the action value for each action at current state.
  arma::colvec actionValue;
  learningNetwork.Predict(state.Encode(), actionValue);

  // Select an action according to the behavior policy.
  action = policy.Sample(actionValue, deterministic, config.NoisyQLearning());
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
    if (config.IsCategorical())
      TrainCategoricalAgent();
    else
      TrainAgent();
  }
  return totalReturn;
}

} // namespace mlpack

#endif
