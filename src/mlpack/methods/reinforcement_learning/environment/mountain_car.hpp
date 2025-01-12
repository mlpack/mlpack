/**
 * @file methods/reinforcement_learning/environment/mountain_car.hpp
 * @author Shangtong Zhang
 *
 * This file is an implementation of Mountain Car task:
 * https://www.gymlibrary.dev/environments/classic_control/mountain_car
 *
 * TODO: provide an option to use dynamics directly from OpenAI gym.
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

/**
 * Implementation of Mountain Car task.
 */
class MountainCar
{
 public:
  /**
   * Implementation of state of Mountain Car. Each state is a
   * (velocity, position) vector.
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State(): data(dimension)
    { /* Nothing to do here. */ }

    /**
     * Construct a state based on the given data.
     *
     * @param data Data for the velocity and position.
     */
    State(const arma::colvec& data): data(data)
    { /* Nothing to do here. */ }

    //! Modify the internal representation of the state.
    arma::colvec& Data() { return data; }

    //! Get the velocity.
    double Velocity() const { return data[0]; }
    //! Modify the velocity.
    double& Velocity() { return data[0]; }

    //! Get the position.
    double Position() const { return data[1]; }
    //! Modify the position.
    double& Position() { return data[1]; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 2;

   private:
    //! Locally-stored velocity and position vector.
    arma::colvec data;
  };

  /**
   * Implementation of action of Mountain Car.
   */
  class Action
  {
   public:
    enum actions
    {
      backward,
      stop,
      forward
    };
    // To store the action.
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 3;
  };

  /**
   * Construct a Mountain Car instance using the given constant.
   *
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   * @param positionMin Minimum legal position.
   * @param positionMax Maximum legal position.
   * @param positionGoal Final target position.
   * @param velocityMin Minimum legal velocity.
   * @param velocityMax Maximum legal velocity.
   * @param doneReward The reward recieved by the agent on success.
   */
  MountainCar(const size_t maxSteps = 200,
              const double positionMin = -1.2,
              const double positionMax = 0.6,
              const double positionGoal = 0.5,
              const double velocityMin = -0.07,
              const double velocityMax = 0.07,
              const double doneReward = 0) :
      maxSteps(maxSteps),
      positionMin(positionMin),
      positionMax(positionMax),
      positionGoal(positionGoal),
      velocityMin(velocityMin),
      velocityMax(velocityMax),
      doneReward(doneReward),
      stepsPerformed(0)
  { /* Nothing to do here */ }

  /**
   * Dynamics of Mountain Car. Get reward and next state based on current state
   * and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return reward, it's always -1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Calculate acceleration.
    int direction = action.action - 1;
    nextState.Velocity() = state.Velocity() + 0.001 * direction - 0.0025 *
        std::cos(3 * state.Position());
    nextState.Velocity() = std::min(std::max(nextState.Velocity(),
        velocityMin), velocityMax);

    // Update states.
    nextState.Position() = state.Position() + nextState.Velocity();
    nextState.Position() = std::min(std::max(nextState.Position(),
        positionMin), positionMax);

    if (nextState.Position() == positionMin && nextState.Velocity() < 0)
      nextState.Velocity() = 0.0;

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward the agent if time ran out.
    if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
      return 0;
    else if (done)
      return doneReward;

    return -1;
  }

  /**
   * Dynamics of Mountain Car. Get reward based on current state and current
   * action.
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, it's always -1.0.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial position is randomly generated within [-0.6, -0.4].
   * Initial velocity is 0.
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    State state;
    stepsPerformed = 0;
    state.Velocity() = 0.0;
    state.Position() = randu() * 0.2 - 0.6;
    return state;
  }

  /**
   * This function checks if the car has reached the terminal state.
   *
   * @param state desired state.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const
  {
    if (maxSteps != 0 && stepsPerformed >= maxSteps)
    {
      Log::Info << "Episode terminated due to the maximum number of steps"
          "being taken.";
      return true;
    }
    else if (state.Position() >= positionGoal)
    {
      Log::Info << "Episode terminated due to agent succeeding.";
      return true;
    }
    return false;
  }

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

 private:
  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored minimum legal position.
  double positionMin;

  //! Locally-stored maximum legal position.
  double positionMax;

  //! Locally-stored goal position.
  double positionGoal;

  //! Locally-stored minimum legal velocity.
  double velocityMin;

  //! Locally-stored maximum legal velocity.
  double velocityMax;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;
};

} // namespace mlpack

#endif
