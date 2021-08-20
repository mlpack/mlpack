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
>::BestAction(const arma::mat& actionValues, const arma::mat& weightSpace)
{
  size_t numWeights = weightSpace.n_cols;
  size_t actionSize = EnvironmentType::Action::size;
  size_t batchSize = actionValues.n_cols / (numWeights * actionSize);
  size_t extendedSize = numWeights * batchSize;

  // Each preference vector is repeated batchSize * actionSize
  // number of times. Shape: (rewardSize, extendedSize * actionSize).
  const arma::mat extWeights = [&]()
  {
    arma::mat retval(rewardSize, extendedSize * actionSize);
    size_t colIdx {}, start {};
    size_t gap = batchSize * actionSize;

    while (colIdx < numWeights)
    {
      retval.submat(arma::span(0, rewardSize),
                    arma::span(start, start + gap - 1)) =
          arma::repmat(weightSpace.col(colIdx), 1, gap);
      start += gap;
      ++colIdx;
    }

    return retval;
  }();

  // Batch multiply extWeights and actionValues and take the max index.
  return arma::index_max(arma::reshape(arma::sum(extWeights % actionValues, 0),
                                       actionSize, extendedSize), 0);
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
  // Begin experience replay. Sample from past experiences.
  // Each input is a unique state-preference pair.
  arma::mat sampledStatePref;
  std::vector<ActionType> sampledActions;
  // Each action results a list of reward.
  arma::mat sampledRewardLists;
  arma::mat sampledNextStatePref;
  arma::irowvec isTerminal;

  // Generate a repository of preference vectors.
  const arma::mat weightSpace =
      arma::normalise(arma::abs(arma::randn(rewardSize, numWeights)), 1, 1);

  learningNetwork.Forward(sampledStatePref, target);
  replayMethod.SampleEQL(sampledStatePref, sampledActions, sampledRewardLists,
      sampledNextStatePref, weightSpace, isTerminal);

  size_t batchSize = sampledStates.n_cols;
  size_t extendedSize = numWeights * batchSize;
  size_t actionSize = sampledActions.n_rows;

  arma::mat nextActionValues(rewardSize, extendedSize * actionSize);
  targetNetwork.Predict(sampledNextStatePref, nextActionValues);

  arma::uvec bestActions{};
  if (config.DoubleQLearning())
  {
    // If use double Q-Learning, use learning network to select the best action.
    arma::mat nextActionValues;
    learningNetwork.Predict(sampledNextStatePref, nextActionValues);
    bestActions = BestAction(nextActionValues);
  }
  else
  {
    bestActions = BestAction(nextActionValues);
  }

  arma::mat target(rewardSize, extendedSize * actionSize);
  learningNetwork.Forward(sampledStatePref, target);

  const double discount = std::pow(config.Discount(), replayMethod.NSteps());

  // Learn from experience.
  arma::mat gradients;
  learningNetwork.BackwardEQL(sampledStatePref, target, gradients, weightSpace);

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
  actionMatrix.resize(ActionType::size, rewardSize);

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
