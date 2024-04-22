/**
 * @file methods/reinforcement_learning/environment/vec_env.hpp
 * @author Ali Hossam
 *
 * This file defines the `VecEnv` class, which serves as a wrapper for
 * vectorized reinforcement learning (RL) environments. It allows for
 * training multiple independent RL environments simultaneously by stacking
 * them into a vector. This approach enables vectorized actions, states,
 * initial states, and rewards.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_VEC_ENV_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_VEC_ENV_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * The `VecEnv` class provides an interface for vectorized RL environments.
 * It wraps multiple instances of an RL environment into a vector to enable
 * training across multiple environments simultaneously.
 *
 * @tparam EnvironmentType The type of RL environment being wrapped.
 */
template <typename EnvironmentType>
class VecEnv
{
 public:
  using EnvState = typename EnvironmentType::State;
  using EnvAction = typename EnvironmentType::Action;

  /**
   * Number of environments defined static so that it is accessed from State and
   * Action classes.
   */
  static size_t nEnvs;

  /**
   * Implementation of the state of the vectorized environment. It is expressed 
   * as a vector of the states of the wrapped environment (EnvironmentType).
   */
  class State
  {
   public:
    State() : states(VecEnv::nEnvs) 
    { /* Nothing to do here. */ }

    /**
     * Constructor with a given vector of states.
     *
     * @param states A vector containing the states.
     */
    State(std::vector<EnvState> states) : states(states) 
    { /* Nothing to do here. */ }

    // Encode the state to a std::vector.
    std::vector<EnvState> Encode() const { return states; }

   private:
    std::vector<EnvState> states;
  };

  /**
   * Implementation of the action of the vectorized environment.
   */
  class Action
  {
   public:
    /** Default constructor. */
    Action() : action(VecEnv::nEnvs) 
    { /* Nothing to do here. */ }

    // A vector of actions for each environment.
    std::vector<EnvAction> action;

    // The size of the action vector.
    static const size_t size = EnvironmentType::Action::size;
  };

  VecEnv()
  {
    for (size_t i = 0; i < nEnvs; i++)
      envs.push_back(EnvironmentType());
  }

  /**
   * Constructor with a vector of environments.
   *
   * @param envs A vector containing the environments.
   */
  VecEnv(const std::vector<EnvironmentType>& envs) : envs(envs)
  {
    nEnvs = envs.size();
  }

  /**
   * Generate initial samples from each environment.
   *
   * @return The initial states of the environments.
   */
  State InitialSample()
  {  
    std::vector<EnvState> initialSamples(nEnvs);
    for (size_t i = 0; i < nEnvs; i++)
      initialSamples[i] = envs[i].InitialSample();
    
    return State(initialSamples);
  }

  /**
   * Sample rewards from each environment given a state and action.
   *
   * @param state The current state of the environments.
   * @param action The action to be taken.
   * @param nextState The next state of the environments.
   *
   * @return A vector containing the rewards from each environment.
   */
  std::vector<double> Sample(const State& state,
                             const Action& action,
                             State& nextState)
  {
    std::vector<double> rewards(nEnvs);
    for (size_t i = 0; i < nEnvs; i++)
      rewards[i] = envs[i].Sample(state.Encode()[i], 
                                   action.action[i],
                                   nextState.Encode()[i]);
    
    return rewards;
  }
  
  /**
   * Sample rewards from each environment given a state and action.
   * The next state is not returned.
   *
   * @param state The current state of the environments.
   * @param action The action to be taken.
   *
   * @return A vector containing the rewards from each environment.
   */
  std::vector<double> Sample(const State& state, const Action& action)
  {
    State nextState(nEnvs);
    return Sample(state, action, nextState);
  }

  /**
   * Check if each environment has reached a terminal state.
   *
   * @param state The current state of the environments.
   *
   * @return A vector indicating whether each environment is in a terminal state.
   */
  std::vector<bool> IsTerminal(const State& state) const
  {
    std::vector<bool> isTerminalVec(nEnvs);
    for (size_t i = 0; i < nEnvs; i++)
      isTerminalVec[i] = envs[i].IsTerminal(state.Encode()[i]);
    
    return isTerminalVec;
  }

 private:
  std::vector<EnvironmentType> envs;

};

} // namespace mlpack
#endif
