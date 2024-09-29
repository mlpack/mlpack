/**
 * @file methods/reinforcement_learning/environment/cont_double_pole_cart.hpp
 * @author Rahul Ganesh Prabhu
 *
 * This file is an implementation of Continuous Double Pole Cart Balancing
 * Task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_CONT_DOUBLE_POLE_CART_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_CONT_DOUBLE_POLE_CART_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of Continuous Double Pole Cart Balancing task. This is an
 * extension of the existing CartPole environment. The environment comprises
 * of a cart with two upright poles of different lengths and masses. The agent
 * is meant to balance the poles by applying force on the cart.
 */
class ContinuousDoublePoleCart
{
 public:
  /**
   * Implementation of the state of Continuous Double Pole Cart. The state is
   * expressed as a vector (position, velocity, angle, angular velocity, angle,
   * angular velocity)
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
     * Construct a state instance from given data.
     *
     * @param data Data for the position, velocity, angle and angular velocity.
     */
    State(const arma::colvec& data) : data(data)
    { /* Nothing to do here */ }

    //! Get the internal representation of the state
    arma::colvec Data() const { return data; }
    //! Modify the internal representation of the state.
    arma::colvec& Data() { return data; }

    //! Get the position of the cart.
    double Position() const { return data[0]; }
    //! Modify the position of the cart.
    double& Position() { return data[0]; }

    //! Get the velocity of the cart.
    double Velocity() const { return data[1]; }
    //! Modify the velocity of the cart.
    double& Velocity() { return data[1]; }

    //! Get the angle of the $i^{th}$ pole.
    double Angle(const size_t i) const { return data[2 * i]; }
    //! Modify the angle of the $i^{th}$ pole.
    double& Angle(const size_t i) { return data[2 * i]; }

    //! Get the angular velocity of the $i^{th}$ pole.
    double AngularVelocity(const size_t i) const { return data[2 * i + 1]; }
    //! Modify the angular velocity of the $i^{th}$ pole.
    double& AngularVelocity(const size_t i) { return data[2 * i + 1]; }

    //! Encode the state to a vector..
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 6;

   private:
    //! Locally-stored state data.
    arma::colvec data;
  };

  /**
   * Implementation of action of Continuous Double Pole Cart.
   */
  struct Action
  {
    double action[1];
    // Storing degree of freedom
    const int size = 1;
  };

  /**
   * Construct a Double Pole Cart instance using the given constants.
   *
   * @param m1 The mass of the first pole.
   * @param m2 The mass of the second pole.
   * @param l1 The length of the first pole.
   * @param l2 The length of the second pole.
   * @param gravity The gravity constant.
   * @param massCart The mass of the cart.
   * @param forceMag The magnitude of the applied force.
   * @param tau The time interval.
   * @param thetaThresholdRadians The maximum angle.
   * @param xThreshold The maximum position.
   * @param doneReward Reward recieved by agent on success.
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   */
  ContinuousDoublePoleCart(const double m1 = 0.1,
                           const double m2 = 0.01,
                           const double l1 = 0.5,
                           const double l2 = 0.05,
                           const double gravity = 9.8,
                           const double massCart = 1.0,
                           const double forceMag = 10.0,
                           const double tau = 0.02,
                           const double thetaThresholdRadians = 36 * 2 *
                              3.1416 / 360,
                           const double xThreshold = 2.4,
                           const double doneReward = 0.0,
                           const size_t maxSteps = 0) :
      m1(m1),
      m2(m2),
      l1(l1),
      l2(l2),
      gravity(gravity),
      massCart(massCart),
      forceMag(forceMag),
      tau(tau),
      thetaThresholdRadians(thetaThresholdRadians),
      xThreshold(xThreshold),
      doneReward(doneReward),
      maxSteps(maxSteps),
      stepsPerformed(0)
  { /* Nothing to do here */ }

  /**
   * Dynamics of Continuous Double Pole Cart instance. Get reward and next
   * state based on current state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState)
  {
    // Update the number of steps performed.
    stepsPerformed++;

    arma::vec dydx(6);
    dydx[0] = state.Velocity();
    dydx[2] = state.AngularVelocity(1);
    dydx[4] = state.AngularVelocity(2);
    Dsdt(state, action, dydx);
    RK4(state, action, dydx, nextState);

    // Check if the episode has terminated.
    bool done = IsTerminal(nextState);

    // Do not reward agent if it failed.
    if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
      return doneReward;
    else if (done)
      return 0;

    /**
     * When done is false, it means that the cartpole has fallen down.
     * For this case the reward is 1.0.
     */
    return 1.0;
  }

  /**
   * This is the ordinary differential equations required for estimation of
   * next state through RK4 method.
   *
   * @param state The current state.
   * @param action The action taken.
   * @param dydx The differential.
   */
  void Dsdt(const State& state,
            const Action& action,
            arma::vec& dydx)
  {
    double totalForce = action.action[0];
    double totalMass = massCart;
    double omega1 = state.AngularVelocity(1);
    double omega2 = state.AngularVelocity(2);
    double sinTheta1 = std::sin(state.Angle(1));
    double sinTheta2 = std::sin(state.Angle(2));
    double cosTheta1 = std::cos(state.Angle(1));
    double cosTheta2 = std::cos(state.Angle(2));

    // Calculate total effective force.
    totalForce += m1 * l1 * omega1 * omega1 * sinTheta1 + 0.375 * m1 * gravity *
        std::sin(2 * state.Angle(1));
    totalForce += m2 * l2 * omega2 * omega2 * sinTheta1 + 0.375 * m2 * gravity *
        std::sin(2 * state.Angle(2));

    // Calculate total effective mass.
    totalMass += m1 * (0.25 + 0.75 * sinTheta1 * sinTheta1);
    totalMass += m2 * (0.25 + 0.75 * sinTheta2 * sinTheta2);

    // Calculate acceleration.
    double xAcc = totalForce / totalMass;
    dydx[1] = xAcc;

    // Calculate angular acceleration.
    dydx[3] = -0.75 * (xAcc * cosTheta1 + gravity * sinTheta1) / l1;
    dydx[5] = -0.75 * (xAcc * cosTheta2 + gravity * sinTheta2) / l2;
  }

  /**
   * This function calls the RK4 iterative method to estimate the next state
   * based on given ordinary differential equation.
   *
   * @param state The current state.
   * @param action The action to be applied.
   * @param dydx The differential.
   * @param nextState The next state.
   */
  void RK4(const State& state,
           const Action& action,
           arma::vec& dydx,
           State& nextState)
  {
    const double hh = tau * 0.5;
    const double h6 = tau / 6;
    arma::vec yt(6);
    arma::vec dyt(6);
    arma::vec dym(6);

    yt = state.Data() + (hh * dydx);
    Dsdt(State(yt), action, dyt);
    dyt[0] = yt[1];
    dyt[2] = yt[3];
    dyt[4] = yt[5];
    yt = state.Data() + (hh * dyt);

    Dsdt(State(yt), action, dym);
    dym[0] = yt[1];
    dym[2] = yt[3];
    dym[4] = yt[5];
    yt = state.Data() + (tau * dym);
    dym += dyt;

    Dsdt(State(yt), action, dyt);
    dyt[0] = yt[1];
    dyt[2] = yt[3];
    dyt[4] = yt[5];
    nextState.Data() = state.Data() + h6 * (dydx + dyt + 2 * dym);
  }

  /**
   * Dynamics of Continuous Double Pole Cart. Get reward based on current
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial state representation is randomly generated within [-0.05, 0.05].
   *
   * @return Initial state for each episode.
   */
  State InitialSample()
  {
    stepsPerformed = 0;
    return State((randu<arma::vec>(6) - 0.5) / 10.0);
  }

  /**
   * This function checks if the car has reached the terminal state.
   *
   * @param state The desired state.
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
    if (std::abs(state.Position()) > xThreshold)
    {
      Log::Info << "Episode terminated due to cart crossing threshold";
      return true;
    }
    if (std::abs(state.Angle(1)) > thetaThresholdRadians ||
        std::abs(state.Angle(2)) > thetaThresholdRadians)
    {
      Log::Info << "Episode terminated due to pole falling";
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
  //! Locally-stored mass of the first pole.
  double m1;

  //! Locally-stored mass of the second pole.
  double m2;

  //! Locally-stored length of the first pole.
  double l1;

  //! Locally-stored length of
  double l2;

  //! Locally-stored gravity.
  double gravity;

  //! Locally-stored mass of the cart.
  double massCart;

  //! Locally-stored magnitude of the applied force.
  double forceMag;

  //! Locally-stored time interval.
  double tau;

  //! Locally-stored maximum angle.
  double thetaThresholdRadians;

  //! Locally-stored maximum position.
  double xThreshold;

  //! Locally-stored done reward.
  double doneReward;

  //! Locally-stored maximum number of steps.
  size_t maxSteps;

  //! Locally-stored number of steps performed.
  size_t stepsPerformed;
};

} // namespace mlpack

#endif
