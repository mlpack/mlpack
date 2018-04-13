/**
 * @file policy_gradient_impl.hpp
 * @author Rohan Raj
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_GENERALISED_PolicyGradient_LEARNING_IMPL_HPP
#define MLPACK_METHODS_RL_GENERALISED_PolicyGradient_LEARNING_IMPL_HPP

#include "policy_gradient.hpp"
namespace mlpack{
namespace rl{
// define the main implementation code
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
PolicyGradient<EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType
>::PolicyGradient(TrainingConfig config,
                  NetworkType network,
                  PolicyType policy,
                  UpdaterType updater,
                  EnvironmentType environment):
    config(std::move(config)),
    learningNetwork(std::move(network)),
    updater(std::move(updater)),
    policy(std::move(policy)),
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
  typename PolicyType
>
double PolicyGradient<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType
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

  // Update current state.
  if (deterministic)
    return reward;

  // Sample from previous experience.
  arma::colvec target(actionValue.n_rows);
  target.zeros();

  // Put action taken by a action as 1.
  target(action) = 1;
  // int target = static_cast<int>(action) + 1;
  advantage = double(totalReturn + reward) - returns.mean();
  // Learn form experience. Experience in form of PolicyGradient.
  arma::mat gradients;
  learningNetwork.Backward(target, gradients);
  gradients *= advantage;
  updater.Update(learningNetwork.Parameters(),
    config.StepSize(), gradients);
  state = nextState;
  return reward;
}
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename BehaviorPolicyType
>
double PolicyGradient<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  BehaviorPolicyType
>::Episode()
{
  // Get the initial state from environment.
  state = environment.InitialSample();
  
  // Calculate total steps in a episode
  steps = 0;

  // Track the return of this episode.
  totalReturn = 0.0;

  // Reset the return status
  returns.reset();

  // Running until get to the terminal state.
  while (!environment.IsTerminal(state))
  {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;

    totalReturn += Step();
    returns(totalReturn);
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
