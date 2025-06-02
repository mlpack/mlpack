/**
 * @file methods/reinforcement_learning/environment/continuous_mountain_car.hpp
 * @author Rohan Raj
 * @author Shashank Shekhar
 *
 * This file is an implementation of Continous Mountain Car task:
 * https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous
 *
 * TODO: provide an option to use dynamics directly from OpenAI gym.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_CONTINUOUS_MOUNTAIN_CAR_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_CONTINUOUS_MOUNTAIN_CAR_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of Continuous Mountain Car task.
 */
class ContinuousMountainCar
{
 public:
  /**
   * Implementation of state of Continuous Mountain Car. Each state is a
   * (velocity, position) vector.
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State() : data(dimension)
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
   * Implementation of action of Continuous Mountain Car.
   * In Continuous mountain car gain, the action represents the
   * force to be applied. This value is bounded in range -1.0 to 1.0.
   * Unlike the simple mountain car environment, where action space has a
   * discrete value, continuous mountain car has continous action space
   * value. 
   */
  struct Action
  {
    double action[1];
    // Storing degree of freedom
    const int size = 1;
  };

  /**
   * Construct a Continuous Mountain Car instance using the given constant.
   *
   * @param positionMin Minimum legal position.
   * @param positionMax Maximum legal position.
   * @param positionGoal Final target position.
   * @param velocityMin Minimum legal velocity.
   * @param velocityMax Maximum legal velocity.
   * @param duration Time Duration for which force is applied on the car.
   * @param doneReward Reward recieved by the agent on success.
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   */
  ContinuousMountainCar(const double positionMin = -1.2,
                        const double positionMax = 0.6,
                        const double positionGoal = 0.45,
                        const double velocityMin = -0.07,
                        const double velocityMax = 0.07,
                        const double duration = 0.0015,
                        const double doneReward = 100,
                        const size_t maxSteps = 0) :
      positionMin(positionMin),
      positionMax(positionMax),
      positionGoal(positionGoal),
      velocityMin(velocityMin),
      velocityMax(velocityMax),
      duration(duration),
      doneReward(doneReward),
      maxSteps(maxSteps),
      stepsPerformed(0)
  { /* Nothing to do here */ }

  /**
   * Dynamics of Continuous Mountain Car. Get reward and next state based 
   * on current state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Calculate acceleration.
    double force = std::min(std::max(action.action[0], -1.0), 1.0);

    // Update states.
    nextState.Velocity() = state.Velocity() + force * duration - 0.0025 *
        std::cos(3 * state.Position());
    nextState.Velocity() = std::min(std::max(nextState.Velocity(),
      velocityMin), velocityMax);
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

    return std::pow(action.action[0], 2) * 0.1;
  }

  /**
   * Dynamics of Continuous Mountain Car. Get reward based on current state and current
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
    state.Position() = Random(-0.6, -0.4);
    return state;
  }

  /**
   * Whether given state is a terminal state.
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

  //! Locally-stored duration.
  double duration;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;
};

} // namespace mlpack

#endif
