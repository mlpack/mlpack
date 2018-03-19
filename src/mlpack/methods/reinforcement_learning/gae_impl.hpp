/**
 * @file gae.hpp
 * @author Rohan Raj
 *
 * Implementation of the Generalised Advantage Estimation method using policy gradient.
 * John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan and Pieter Abbeel
 * HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION, ICLR 2016
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_GENERALISED_ADVANTAGE_LEARNING_IMPL_HPP
#define MLPACK_METHODS_RL_GENERALISED_ADVANTAGE_LEARNING_IMPL_HPP

#include "gae.hpp"
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
Advantage<EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Advantage(TrainingConfig config,
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
    deterministic(false),
    steps(0)
{
  if (learningNetwork.Parameters().is_empty())
    learningNetwork.ResetParameters();
  this->updater.Initialize(learningNetwork.Parameters().n_rows,
      learningNetwork.Parameters().n_cols);
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double Advantage<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
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
  replayMethod.StoreEpisode(state, action, reward,
      nextState, environment.IsTerminal(state), config.Discount());

  // Update current state.

  if (deterministic || totalSteps < config.ExplorationSteps())
    return reward;

  // Start experience replay.

  // Sample from previous experience.
  if (environment.IsTerminal(nextState) || steps == config.StepLimit())
  {
      arma::mat sampledStates;
      arma::icolvec sampledActions;
      arma::colvec advantage;
      arma::icolvec isTerminal;
      replayMethod.EpisodeReplay(sampledStates, sampledActions, advantage,
      isTerminal);
      // Compute action value for next state with target network.

	  /***One way to get number of action is by making a variable capturing the number of action
	  * Another way is to use the predict funtion to get the size of the last layer.
	  * 
	  * First way will make the program generic for any number of action space.
	  * Second way is easy. But it is unnecessarily increase the computation.
	  *
	  * I am currently taking inputs from feed forward network and putting it to zero.
	  * This way I am meeting my aim of using advantage.
	  *
	  * I have also defined policygradient loss in mlpack/methods/ann/layers for implementation.
	  */
      arma::mat target;
      learningNetwork.Forward(sampledStates, target);
      /* turning the target to zero to place advantage value
       * in place of it
       */
      target.zeros();
      for (size_t i = 0; i < sampledStates.n_cols; ++i)
      {
        target(sampledActions[i], i) = advantage[i]; // learning from advantage
      }

      // Learn form experience. Experience in form of advantage.
      arma::mat gradients;
      learningNetwork.Backward(target, gradients);
      updater.Update(learningNetwork.Parameters(),
        config.StepSize(), gradients);
  }
  state = nextState;
  return reward;
}
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType,
  typename ReplayType
>
double Advantage<
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
  steps = 0;

  // Track the return of this episode.
  double totalReturn = 0.0;

  // Running until get to the terminal state.
  while (!environment.IsTerminal(state))
  {
    if (config.StepLimit() && steps > config.StepLimit())
      break;

    totalReturn += Step();
    steps++;

    if (deterministic)
      continue;

    totalSteps++;
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack

#endif
