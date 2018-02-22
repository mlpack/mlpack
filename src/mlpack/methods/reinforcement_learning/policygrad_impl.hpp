#ifndef MLPACK_METHODS_RL_POLICY_GRADIENT_LEARNING_IMPL_HPP
#define MLPACK_METHODS_RL_POLICY_GRADIENT_LEARNING_IMPL_HPP

#include "policygrad.hpp"
namespace mlpack{
namespace rl{
// define the main implementation code
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
PolicyGradient<EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::PolicyGradient(TrainingConfig config,
             NetworkType network,
             PolicyType policy,
             ReplayType replayMethod,
             UpdaterType updater,
             EnvironmentType environment):
    config(std::move(config)),
    learningNetwork(std::move(network)),
    updater(std::move(updater)),
    policy(std::move(policy)),
    replayMethod(std::move(replayMethod)),
    environment(std::move(environment)),
    totalSteps(0),
    deterministic(false)
{
  if (learningNetwork.Parameters().is_empty())
    learningNetwork.ResetParameters();
  this->updater.Initialize(learningNetwork.Parameters().n_rows,
      learningNetwork.Parameters().n_cols);
  targetNetwork = learningNetwork;
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double PolicyGradient<
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
  arma::colvec advantage;
  arma::icolvec isTerminal;
  replayMethod.PolicySample(sampledStates, sampledActions, advantage, isTerminal); //rohan
  // Compute action value for next state with target network.

  arma::Mat<size_t> target(2,sampledStates.n_cols)  ; // 2 is the action size
  target.zeros(); // ACTION SIZE
  for (size_t i = 0; i < sampledStates.n_cols; ++i)
  {
  	target(sampledActions[i], i) = advantage[i]; // learning from advantage
  }

  // Learn form experience. Experience in form of advantage.
  arma::mat gradients;
  learningNetwork.Backward(target, gradients);
  updater.Update(learningNetwork.Parameters(), config.StepSize(), gradients);

  return reward;
}
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double PolicyGradient<
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

}
}

#endif