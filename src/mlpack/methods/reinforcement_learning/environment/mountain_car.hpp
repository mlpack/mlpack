/**
 * @file mountain_car.hpp
 * @author Shangtong Zhang
 *
 * This file is an implementation of Mountain Car task
 * https://gym.openai.com/envs/MountainCar-v0
 *
 * TODO: refactor to OpenAI interface
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_MOUNTAIN_CAR_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_MOUNTAIN_CAR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of Mountain Car task
 */
class MountainCar
{
 public:

  /**
   * Implementation of state of Mountain Car
   * Each state is a (velocity, position) pair
   */
  class State
  {
   public:
    //! Construct a state instance
    State(): data(2, arma::fill::zeros) { }

    /**
     * Construct a state based on given data
     * @param data desired internal data
     */
    State(const arma::colvec& data): data(data) { }

    //! Encode the state to a column vector
    const arma::colvec& Encode() const { return data; }

    /**
     * Set the internal data to given value
     * @param data desired internal data
     */
    void Set(const arma::colvec& data) { this->data = data; }

    //! Get velocity
    double Velocity() const { return data[0]; }

    //! Modify velocity
    double& Velocity() { return data[0]; }

    //! Get position
    double Position() const { return data[1]; }

    //! Modify position
    double& Position() { return data[1]; }

   private:
    //! Locally-stored velocity and position
    arma::colvec data;
  };

  /**
   * Implementation of action of Mountain Car
   */
  enum Action
  {
    backward,
    stop,
    forward,

    // Track the size of the action space
    size
  };

  /**
   * Construct a Mountain Car instance
   * @param positionMin minimum legal position
   * @param positionMax maximum legal position
   * @param velocityMin minimum legal velocity
   * @param velocityMax maximum legal velocity
   */
  MountainCar(double positionMin = -1.2, double positionMax = 0.5, double velocityMin = -0.07, double velocityMax = 0.07):
      positionMin(positionMin), positionMax(positionMax), velocityMin(velocityMin), velocityMax(velocityMax) { }

  /**
   * Dynamics of Mountain Car
   * Get reward and next state based on current state and current action
   * @param state Current state
   * @param action Current action
   * @param nextState Next state
   * @return reward, it's always -1.0
   */
  double Sample(const State& state, const Action& action, State& nextState) const
  {
    int direction = action - 1;
    nextState.Velocity() = state.Velocity() + 0.001 * direction - 0.0025 * std::cos(3 * state.Position());
    nextState.Velocity() = std::min(std::max(nextState.Velocity(), velocityMin), velocityMax);

    nextState.Position() = state.Position() + nextState.Velocity();
    nextState.Position() = std::min(std::max(nextState.Position(), positionMin), positionMax);

    if (std::abs(nextState.Position() - positionMin) <= 1e-5)
    {
      nextState.Velocity() = 0.0;
    }

    return -1.0;
  }

  /**
   * Dynamics of Mountain Car
   * Get reward based on current state and current action
   * @param state Current state
   * @param action Current action
   * @return reward, it's always -1.0
   */
  double Sample(const State& state, const Action& action) const
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial position is randomly generated within [-0.6, -0.4]
   * Initial velocity is 0
   * @return Initial state for each episode
   */
  State InitialSample() const
  {
    State state;
    state.Velocity() = 0.0;
    state.Position() = arma::as_scalar(arma::randu(1)) * 0.2 - 0.6;
    return state;
  }

  /**
   * Whether given state is terminal state
   * @param state desired state
   * @return true if @state is terminal state, otherwise false
   */
  bool IsTerminal(const State& state) const { return std::abs(state.Position() - positionMax) <= 1e-5; }

 private:
  //! Locally-stored minimum legal position
  double positionMin;

  //! Locally-stored maximum legal position
  double positionMax;

  //! Locally-stored minimum legal velocity
  double velocityMin;

  //! Locally-stored maximum legal velocity
  double velocityMax;

};

} // namespace rl
} // namespace mlpack

#endif
