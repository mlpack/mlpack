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
       PolicyType& actionPolicy,
       ReplayType& replayMethod,
       UpdaterType updater,
       EnvironmentType environment):
    config(config),
    learningNetwork(network),
    actionPolicy(actionPolicy),
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
  typename ActionPolicyType,
  typename LambdaUpdatePolicyType,
  typename ReplayType
>
EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  ActionPolicyType,
  LambdaUpdatePolicyType,
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
  typename ActionPolicyType,
  typename LambdaUpdatePolicyType,
  typename ReplayType
>
arma::uvec EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  ActionPolicyType,
  LambdaUpdatePolicyType,
  ReplayType
>::BestAction(const arma::mat& actionValues, const arma::mat& extendedWeightSpace)
{
  // Perform batch dot product between extWeights and actionValues
  // followed by storing the index of max elements.
  return arma::index_max(arma::reshape(
      arma::sum(extendedWeightSpace % actionValues),
      EnvironmentType::Action::size,                             // Action size.
      extendedWeightSpace.n_cols / EnvironmentType::Action::size // Input size.
      ));
};

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename ActionPolicyType,
  typename LambdaUpdatePolicyType,
  typename ReplayType
>
void EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  ActionPolicyType,
  LambdaUpdatePolicyType,
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


  size_t batchSize = sampledStates.n_cols;.
  // Count of total state-preference pairs.
  size_t inputSize = batchSize * numWeights;

  // Generate a repository of preference vectors.
  const arma::mat weightSpace =
      arma::normalise(arma::abs(arma::randn(EnvironmentType::rewardSize, numWeights)), 1, 1);

  // Each preference vector is repeated batchSize * actionSize
  // number of times. Shape: (rewardSize, inputSize * actionSize).
  const arma::mat extendedWeightSpace = [batchSize, inputSize, &weightSpace]()
  {
    arma::mat retval(EnvironmentType::rewardSize, inputSize * actionSize);
    size_t colIdx = 0, start = 0;
    const size_t gap = batchSize * actionSize;

    while (colIdx < numWeights)
    {
      retval.submat(arma::span(0, EnvironmentType::rewardSize),
                    arma::span(start, start + gap - 1)) =
          arma::repmat(weightSpace.col(colIdx), 1, gap);
      start += gap;
      ++colIdx;
    }

    return retval;
  }();

  learningNetwork.Forward(sampledStatePref, target);
  replayMethod.SampleEQL(sampledStatePref, sampledActions, sampledRewardLists,
      sampledNextStatePref, weightSpace, isTerminal);

  // For each state-preference pair, the target network outputs
  // actionSize number of reward vectors.
  arma::mat nextActionValues(EnvironmentType::rewardSize, inputSize * actionSize);
  targetNetwork.Predict(sampledNextStatePref, nextActionValues);

  arma::uvec bestActions{};
  if (config.DoubleQLearning())
  {
    // If use double Q-Learning, use learning network to select the best action.
    arma::mat nextActionValues;
    learningNetwork.Predict(sampledNextStatePref, nextActionValues);
    bestActions = BestAction(nextActionValues, extendedWeightSpace);
  }
  else
  {
    bestActions = BestAction(nextActionValues, extendedWeightSpace);
  }

  arma::mat target(EnvironmentType::rewardSize, inputSize * actionSize);
  learningNetwork.Forward(sampledStatePref, target);

  const double discount = std::pow(config.Discount(), replayMethod.NSteps());

  // Each slice of the cube holds the action vectors of a state-preference pair.
  arma::cube targetCube(target.memptr(), EnvironmentType::rewardSize, actionSize,
                        inputSize, false, true);
  arma::cube nextActionValCube(nextActionValues.memptr(), EnvironmentType::rewardSize,
                               actionSize, inputSize, false, true);

  // Iterate over state-preference indexes (spIdx).
  for (size_t spIdx = 0; spIdx < inputSize; ++spIdx)
  {
    // The multi-objective Bellman filter (H) applied over nextActionValue.
    const arma::vec& hQ = nextActionValCube.slice(spIdx).col(bestAction(spIdx));

    targetCube.slice(spIdx).col(sampledActions(spIdx).action) =
        sampledRewardLists(spIdx) + discount * hQ * (1 - isTerminal(spIdx));
  }

  // Learn from experience.
  arma::mat gradients;
  learningNetwork.Backward(
      sampledStatePref, target, extendedWeightSpace,
      [&lambdaUpdatePolicy.Lambda(), &extendedWeightSpace](
          const arma::mat &predictions, const arma::mat &targets)
      {
        const size_t numElem = arma::sum((predictions - targets) != 0);
        const double lossA =
            std::pow(arma::norm((predictions - targets).vectorise()), 2) /
            numElem;
        const double lossB =
            std::pow(arma::norm(arma::sum(extendedWeightSpace %
                                          (predictions - targets))),
                     2) /
            numElem;

        const double homotopyLoss = (1 - lambda) * lossA + lambda * lossB;

        // Store the error.
        arma::mat errorA = (predictions - targets) / numElem;
        arma::mat errorB =
            arma::sum(extendedWeightSpace % (predictions - targets)) %
            extendedWeightSpace;
        const double error = 2 * ((1 - lambda) * errorA + lambda * errorB);
        return std::make_tuple(error, homotopyLoss);
      },
      gradients);

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
  {
    actionPolicy.Anneal();
    lambdaUpdatePolicy.Anneal();
  }
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename ActionPolicyType,
  typename LambdaUpdatePolicyType,
  typename ReplayType
>
void EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  ActionPolicyType,
  LambdaUpdatePolicyType,
  ReplayType
>::SelectAction()
{
  // Stores the Q vector of each action. Shape: (rewardSize, actionSize).
  arma::mat actionValueMatrix;
  learningNetwork.Predict(arma::join_cols(state, preference), actionValueMatrix);
  arma::inplace_trans(actionValueMatrix);
  arma::vec utilityValue = actionValueMatrix * preference;
  // Select an action according to the behavior policy.
  action = actionPolicy.Sample(utilityValue, deterministic, config.NoisyEQL());
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename ActionPolicyType,
  typename LambdaUpdatePolicyType,
  typename ReplayType
>
void EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  ActionPolicyType,
  LambdaUpdatePolicyType,
  ReplayType
>::AgentReset()
{
  preference.reset();
  lambdaUpdatePolicy.Anneal();
  actionPolicy.Anneal();
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename ActionPolicyType,
  typename LambdaUpdatePolicyType,
  typename ReplayType
>
double EQL<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  ActionPolicyType,
  LambdaUpdatePolicyType,
  ReplayType
>::Episode()
{
  // Get the initial state from environment.
  state = environment.InitialSample();
  preference = arma::normalise(arma::abs(arma::randn(EnvironmentType::rewardSize, 1)), 1);
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

    if (totalSteps > config.StepLimit())
    {
      AgentReset();
      break;
    }
  }
  return totalReturn;
}

} // namespace rl
} // namespace mlpack

#endif
