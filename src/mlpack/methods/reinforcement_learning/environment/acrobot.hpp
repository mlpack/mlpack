/**
 * @file methods/reinforcement_learning/environment/acrobot.hpp
 * @author Rohan Raj
 *
 * This file is an implementation of Acrobot task:
 * https://www.gymlibrary.dev/environments/classic_control/acrobot
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_ENVIRONMENT_ACROBOT_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_ACROBOT_HPP

#include <mlpack/core.hpp>

namespace mlpack{

/**
 * Implementation of Acrobot game. Acrobot is a 2-link pendulum with only the
 * second joint actuated. Intitially, both links point downwards. The goal is
 * to swing the end-effector at a height at least the length of one link above
 * the base. Both links can swing freely and can pass by each other, i.e.,
 * they don't collide when they have the same angle.
 */
class Acrobot
{
 public:
  /*
   * Implementation of Acrobot State. Each State is a tuple vector
   * (theta1, thetha2, angular velocity 1, angular velocity 2).
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
     * @param data Data for the theta and angular velocity of two links.
     */
    State(const arma::colvec& data) : data(data)
    { /* nothing to do here */ }

    //! Modify the state representation.
    arma::colvec& Data() { return data; }

    //! Get value of theta (one).
    double Theta1() const { return data[0]; }
    //! Modify value of theta (one).
    double& Theta1() { return data[0]; }

    //! Get value of theta (two).
    double Theta2() const { return data[1]; }
    //! Modify value of theta (two).
    double& Theta2() { return data[1]; }

    //! Get value of Angular velocity (one).
    double AngularVelocity1() const { return data[2]; }
    //! Modify the angular velocity (one).
    double& AngularVelocity1() { return data[2]; }

    //! Get value of Angular velocity (two).
    double AngularVelocity2() const { return data[3]; }
    //! Modify the angular velocity (two).
    double& AngularVelocity2() { return data[3]; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 4;

   private:
    //! Locally-Stored (theta1, theta2, angular velocity 1, angular velocity2).
    arma::colvec data;
  };

  /*
   * Implementation of action for Acrobot
   */
  class Action
  {
   public:
    enum actions
    {
      negativeTorque,
      zeroTorque,
      positiveTorque,
    };
    // To store the action.
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 3;
  };

  /**
   * Construct a Acrobot instance using the given constants.
   *
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
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
   */
  Acrobot(const size_t maxSteps = 500,
          const double gravity = 9.81,
          const double linkLength1 = 1.0,
          const double linkLength2 = 1.0,
          const double linkMass1 = 1.0,
          const double linkMass2 = 1.0,
          const double linkCom1 = 0.5,
          const double linkCom2 = 0.5,
          const double linkMoi = 1.0,
          const double maxVel1 = 4 * M_PI,
          const double maxVel2 = 9 * M_PI,
          const double dt = 0.2,
          const double doneReward = 0) :
      maxSteps(maxSteps),
      gravity(gravity),
      linkLength1(linkLength1),
      linkLength2(linkLength2),
      linkMass1(linkMass1),
      linkMass2(linkMass2),
      linkCom1(linkCom1),
      linkCom2(linkCom2),
      linkMoi(linkMoi),
      maxVel1(maxVel1),
      maxVel2(maxVel2),
      dt(dt),
      doneReward(doneReward),
      stepsPerformed(0)
  { /* Nothing to do here */ }

  /**
   * Dynamics of the Acrobot System. To get reward and next state based on
   * current state and current action. Always return -1 reward.
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
    arma::colvec currentState = {state.Theta1(), state.Theta2(),
        state.AngularVelocity1(), state.AngularVelocity2()};

    arma::colvec currentNextState = Rk4(currentState, Torque(action));

    nextState.Theta1() = Wrap(currentNextState[0], -M_PI, M_PI);

    nextState.Theta2() = Wrap(currentNextState[1], -M_PI, M_PI);

    //! The value of angular velocity is bounded in min and max value.
    nextState.AngularVelocity1() = std::min(std::max(currentNextState[2],
        -maxVel1), maxVel1);
    nextState.AngularVelocity2() = std::min(std::max(currentNextState[3],
        -maxVel2), maxVel2);

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward the agent if time ran out.
    if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
      return 0;
    else if (done)
      return doneReward;

    return -1;
  };

  /**
   * Dynamics of the Acrobot System. To get reward and next state based on
   * current state and current action. This function calls the Sample function
   * to estimate the next state return reward for taking a particular action.
   *
   * @param state The current State.
   * @param action The action taken.
   * @return nextState The next state.
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
    return State((randu<arma::colvec>(4) - 0.5) / 5.0);
  }

  /**
   * This function checks if the acrobot has reached the terminal state.
   *
   * @param state The current State.
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
    else if (-std::cos(state.Theta1()) - std::cos(state.Theta1() +
        state.Theta2()) > 1.0)
    {
      Log::Info << "Episode terminated due to agent succeeding.";
      return true;
    }
    return false;
  }

  /**
   * This is the ordinary differential equations required for estimation of
   * nextState through RK4 method.
   *
   * @param state Current State.
   * @param torque The torque Applied.
   */
  arma::colvec Dsdt(arma::colvec state, const double torque) const
  {
    const double m1 = linkMass1;
    const double m2 = linkMass2;
    const double l1 = linkLength1;
    const double lc1 = linkCom1;
    const double lc2 = linkCom2;
    const double I1 = linkMoi;
    const double I2 = linkMoi;
    const double g = gravity;
    const double a = torque;
    const double theta1 = state[0];
    const double theta2 = state[1];

    arma::colvec values(4);
    values[0] = state[2];
    values[1] = state[3];

    const double d1 = m1 * std::pow(lc1, 2) + m2 * (std::pow(l1, 2) +
        std::pow(lc2, 2) + 2 * l1 * lc2 * std::cos(theta2)) + I1 + I2;

    const double d2 = m2 * (std::pow(lc2, 2) + l1 * lc2 * std::cos(theta2)) +
        I2;

    const double phi2 = m2 * lc2 * g * std::cos(theta1 + theta2 - M_PI / 2.);

    const double phi1 = - m2 * l1 * lc2 * std::pow(values[1], 2) *
        std::sin(theta2) - 2 * m2 * l1 * lc2 * values[1] * values[0] *
        std::sin(theta2) + (m1 * lc1 +  m2 * l1) * g *
        std::cos(theta1 - M_PI / 2) + phi2;

    values[3] = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * std::pow(values[0], 2) *
        std::sin(theta2) - phi2) / (m2 * std::pow(lc2, 2) + I2 -
        std::pow(d2, 2) / d1);

    values[2] = -(d2 * values[3] + phi1) / d1;

    return values;
  };

  /**
   * Wrap funtion is required to truncate the angle value from -180 to 180.
   * This function will make sure that value will always be between minimum
   * to maximum.
   *
   * @param value Scalar value to wrap.
   * @param minimum Minimum range of wrap.
   * @param maximum Maximum range of wrap.
   */
  double Wrap(double value,
              const double minimum,
              const double maximum) const
  {
    const double diff = maximum - minimum;

    if (value > maximum)
    {
      value = value - diff;
    }
    else if (value < minimum)
    {
      value = value + diff;
    }

    return value;
  };

  /**
   * This function calculates the torque for a particular action.
   * 0 : negative torque, 1 : zero torque, 2 : positive torque.
   *
   * @param action Action taken.
   */
  double Torque(const Action& action) const
  {
    // Add noise to the Torque Torque is action number - 1. {0,1,2} -> {-1,0,1}.
    return double(action.action - 1) + Random(-0.1, 0.1);
  }

  /**
   * This function calls the RK4 iterative method to estimate the next state
   * based on given ordinary differential equation.
   *
   * @param state The current State.
   * @param torque The torque applied.
   */
  arma::colvec Rk4(const arma::colvec state, const double torque) const
  {
    arma::colvec k1 = Dsdt(state, torque);
    arma::colvec k2 = Dsdt(state + dt * k1 / 2, torque);
    arma::colvec k3 = Dsdt(state + dt * k2 / 2, torque);
    arma::colvec k4 = Dsdt(state + dt * k3, torque);
    arma::colvec nextState = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    return nextState;
  };

  //! Get the number of steps performed.
  size_t StepsPerformed() const { return stepsPerformed; }

  //! Get the maximum number of steps allowed.
  size_t MaxSteps() const { return maxSteps; }
  //! Set the maximum number of steps allowed.
  size_t& MaxSteps() { return maxSteps; }

 private:
  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored gravity.
  double gravity;

  //! Locally-stored length of link 1.
  double linkLength1;

  //! Locally-stored length of link 2.
  double linkLength2;

  //! Locally-stored mass of link 1.
  double linkMass1;

  //! Locally-stored mass of link 2.
  double linkMass2;

  //! Locally-stored position of link 1.
  double linkCom1;

  //! Locally-stored position of link 2.
  double linkCom2;

  //! Locally-stored moment of intertia value.
  double linkMoi;

  //! Locally-stored max angular velocity of link1.
  double maxVel1;

  //! Locally-stored max angular velocity of link2.
  double maxVel2;

  //! Locally-stored dt for RK4 method.
  double dt;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;
};

} // namespace mlpack

#endif
