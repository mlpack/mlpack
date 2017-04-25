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

namespace mountain_car_details {
constexpr double positionMin = -1.2;
constexpr double positionMax = 0.5;
constexpr double velocityMin = -0.07;
constexpr double velocityMax = 0.07;
}

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
    State(double velocity = 0, double position = 0) : data(2)
    {
      this->Velocity() = velocity;
      this->Position() = position;
    }

    //! Encode the state to a column vector
    const arma::colvec& Encode() const
    {
      return data;
    }

    //! Get velocity
    double Velocity() const
    {
      return data[0];
    }

    //! Modify velocity
    double& Velocity()
    {
      return data[0];
    }

    //! Get position
    double Position() const
    {
      return data[1];
    }

    //! Modify position
    double& Position()
    {
      return data[1];
    }

    //! Whether current state is terminal state
    bool IsTerminal() const
    {
      using namespace mountain_car_details;
      return std::abs(Position() - positionMax) <= 1e-5;
    }

   private:
    //! Locally-stored velocity and position
    arma::colvec data;
  };

  /**
   * Implementation of action of Mountain Car
   */
  class Action
  {
   public:
    enum Actions
    {
      backward,
      stop,
      forward
    };

    //! # of actions
    static constexpr size_t count = 3;
  };

  /**
   * Dynamics of Mountain Car
   * Get next state and next action based on current state and current action
   * @param state Current state
   * @param action Current action
   * @param nextState Next state
   * @param reward Reward is always -1
   */
  void Sample(const State& state, const Action::Actions& action,
              State& nextState, double& reward)
  {
    using namespace mountain_car_details;
    int direction = action - 1;
    nextState.Velocity() = state.Velocity() + 0.001 * direction - 0.0025 * std::cos(3 * state.Position());
    nextState.Velocity() = std::min(std::max(nextState.Velocity(), velocityMin), velocityMax);

    nextState.Position() = state.Position() + nextState.Velocity();
    nextState.Position() = std::min(std::max(nextState.Position(), positionMin), positionMax);

    reward = -1.0;
    if (std::abs(nextState.Position() - positionMin) <= 1e-5)
    {
      nextState.Velocity() = 0.0;
    }
  }

  /**
   * Initial position is randomly generated within [-0.6, -0.4]
   * Initial velocity is 0
   * @return Initial state for each episode
   */
  State InitialSample()
  {
    State state;
    state.Velocity() = 0.0;
    state.Position() = arma::as_scalar(arma::randu(1)) * 0.2 - 0.6;
    return state;
  }

};

} // namespace rl
} // namespace mlpack

#endif
