/**
 * @file lunarlander.hpp
 * @author Xiaohong Ji
 *
 * This file is an implementation of Acrobot task:
 * https://github.com/zippkidd/lunarLander
 * https://gym.openai.com/envs/LunarLanderContinuous-v2/
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_LUNARLANDER_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_LUNARLANDER_HPP

#include <mlpack/core.hpp>

namespace mlpack{
namespace rl{

/**
 * Implementation of Lunarlander game. Lunarlander is a 2-link pendulum with only the
 * second joint actuated. Intitially, both links point downwards. The goal is
 * to swing the end-effector at a height at least the length of one link above
 * the base. Both links can swing freely and can pass by each other, i.e.,
 * they don't collide when they have the same angle.
 */
class LunarLander
{
 public:
  /**
   * Implementation of LunarLander State. Each State is a tuple vector
   * (Time, Height, Speed, Fuel).
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State(): data(dimension) { /* nothing to do here */ }

    /**
     * Construct a state instance from given data.
     *
     * @param data Data for (Time, Height, Speed, Fuel).
     */
    State(const arma::colvec& data) : data(data)
    { /* nothing to do here */ }

    //! Modify the state representation.
    arma::colvec& Data() { return data; }

    //! Get value of time.
    double ElapsedTime() const { return data[0]; }
    //! Modify value of time.
    double& ElapsedTime() { return data[0]; }

    //! Get value of height.
    double Height() const { return data[1]; }
    //! Modify value of height.
    double& Height() { return data[1]; }

    //! Get value of speed.
    double Speed() const { return data[2]; }
    //! Modify the speed.
    double& Speed() { return data[2]; }

    //! Get value of Fuel.
    double Fuel() const { return data[3]; }
    //! Modify the Fuel.
    double& Fuel() { return data[3]; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 4;

   private:
    //! Locally-Stored (Time, Height, Speed, Fuel).
    arma::colvec data;
  };

  /*
   * Implementation of action for Lunarlander
   */
  struct Action
  {
    // Fuel burn amount
    double action = 0.0;
    // Storing degree of freedom
    int size = 1;
  };

  /**
   * Construct a Acrobot instance using the given constants.
   *
   * @param gravity The gravity parameter.
   * @param linkLength1 The length of link 1.
   * @param linkLength2 The length of link 2.
   * @param linkMass1 The mass of link 1.
   * @param linkMass2 The mass of link 2.
   * @param linkCom1 The position of the center of mass of link 1.
   * @param linkCom2 The position of the center of mass of link 2.
   * @param linkMoi The moments of inertia for both links.
   * @param maxVel1 The max angular velocity of link1.
   * @param maxVel2 The max angular velocity of link2.
   * @param dt The differential value.
   * @param doneReward The reward recieved by the agent on success.
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   */
  LunarLander(const double elapsedTime = 0,
              const double velocity = 50,
              const double burnAmount = 0,
              const double height = 1000,
              const double fuelRemaining = 150,
              const double doneReward = 200) :
      elapsedTime(elapsedTime),
      velocity(velocity),
      burnAmount(burnAmount),
      height(height),
      fuelRemaining(fuelRemaining)
  { /* Nothing to do here */ }

  arma::colvec UpdateStatus(const State& state,
                    const Action& action)
  {
    double burnAmount = action.action;
    double fuelRemaining = state[3] - burnAmount;
    double height = state[1] - (((state[2] - burnAmount) + 5) + state[2]) / 2);
    double velocity = state[2] - burnAmount + 5;
    return {0, height, velocity, fuelRemaining};
  }

  /**
   * Dynamics of the LunarLander System. To get reward and next state based on
   * current state and current action.
   *
   * @param state The current State.
   * @param action The action taken.
   * @param nextState The next state.
   * @return reward, it's always -1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    // Make a vector to estimate nextstate.
    arma::colvec currentState = {state.ElapsedTime(), state.Height(),
        state.Speed(), state.Fuel()};

    arma::colvec currentNextState = UpdateStatus(currentState, action);

    nextState.ElapsedTime() = currentNextState.ElapsedTime();
    nextState.Height() = currentNextState.Height();
    nextState.Speed() = currentNextState.Speed();
    nextState.Fuel() = std::max(0, currentNextState.Fuel());

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward the agent if time ran out.
    if (done && nextState.Speed() < 0)
      return 0;
    else if (done && nextState.Speed() == 0)
      return doneReward;

    return -1;
  };

  /**
   * Dynamics of the LunarLander System. To get reward and next state based on
   * current state and current action. This function calls the Sample function
   * to estimate the next state return reward for taking a particular action.
   *
   * @param state The current State.
   * @param action The action taken.
   * @param nextState The next state.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * This function does random initialization of state space.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    return State({elapsedTime, height, velocity, fuelRemaining});
  }

  /**
   * This function checks if the acrobot has reached the terminal state.
   *
   * @param state The current State.
   * @return true if state is a terminal state, otherwise false.
   */
  bool IsTerminal(const State& state) const {
    if (state.Fuel() <= 0) return true;
    if (state.Speed() <= 0) return true;
    return false;
  }

 private:
  double elapsedTime;

  double velocity;

  double burnAmount;

  double height;

  double fuelRemaining;

  double doneReward;

  size_t stepsPerformed;
};

} // namespace rl
} // namespace mlpack

#endif
