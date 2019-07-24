/**
 * @file continuous_multiple_pole_cart.hpp
 * @author Rahul Ganesh Prabhu
 *
 * This file is an implementation of Continuous Multiple Pole Cart Balancing
 * Task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_CONTINUOUS_MULTIPLE_POLE_CART_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_CONTINUOUS_MULTIPLE_POLE_CART_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of Continuous Multiple Pole Cart Balancing task.
 */
class ContinuousMultiplePoleCart
{
 public:
  /**
   * Implementation of the state of Continuous Multiple Pole Cart. The state is expressed as
   * a matrix where the $0^{th}$ column is the state of the cart, represented by a tuple
   * (position, velocity) and the $i^{th}$ column is the state of the $i^{th}$ pole, represented
   * by a tuple (angle, angular velocity).
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     * 
     * @param numPoles The number of poles.
     */
    State(const size_t numPoles)
    {
      data = arma::zeros<arma::mat>(dimension, numPoles + 1);
    }

    /**
     * Construct a state instance from given data.
     *
     * @param data Data for the position, velocity, angle and angular velocity.
     */
    State(const arma::mat& data) : data(data)
    { /* Nothing to do here */ }

    //! Modify the internal representation of the state.
    arma::mat& Data() { return data; }

    //! Get the position of the cart.
    double Position() const { return data(0, 0); }
    //! Modify the position of the cart.
    double& Position() { return data(0, 0); }

    //! Get the velocity of the cart.
    double Velocity() const { return data(1, 0); }
    //! Modify the velocity of the cart.
    double& Velocity() { return data(1, 0); }

    //! Get the angle of the $i^{th}$ pole with the vertical.
    double Angle(const size_t i) const { return data(0, i); }
    //! Modify the angle of the $i^{th}$ pole with the vertical.
    double& Angle(const size_t i) { return data(0, i); }

    //! Get the angular velocity of the $i^{th}$ pole.
    double AngularVelocity(const size_t i) const { return data(1, i); }
    //! Modify the angular velocity of the $i^{th}$ pole.
    double& AngularVelocity(const size_t i) { return data(1, i); }

    //! Encode the state to a matrix.
    const arma::mat& Encode() const { return data; }

    //! Dimension of the encoded state.
    const size_t dimension = 2;

   private:
    //! Locally-stored state data.
    arma::mat data;
  };

  /**
   * Implementation of action of Continuous Multiple Pole Cart.
   */
  struct Action
  {
    double action = 0.0;
    // Track the size of the action space.
    const int size = 1;
  };

  /**
   * Construct a Multiple Pole Cart instance using the given constants.
   *
   * @param poleNum The number of poles
   * @param gravity The gravity constant.
   * @param massCart The mass of the cart.
   * @param massPole The mass of the pole.
   * @param length The length of the pole.
   * @param tau The time interval.
   * @param thetaThresholdRadians The maximum angle.
   * @param xThreshold The maximum position.
   * @param doneReward Reward recieved by agent on success.
   * @param maxSteps The number of steps after which the episode
   *    terminates. If the value is 0, there is no limit.
   */
  ContinuousMultiplePoleCart(const size_t poleNum,
                             const arma::vec& poleLengths,
                             const arma::vec& poleMasses,
                             const double gravity = 9.8,
                             const double massCart = 1.0,
                             const double tau = 0.02,
                             const double thetaThresholdRadians = 12 * 2 *
                                3.1416 / 360,
                             const double xThreshold = 2.4,
                             const double doneReward = 0.0,
                             const size_t maxSteps = 0) :
      poleNum(poleNum),
      poleLengths(poleLengths),
      poleMasses(poleMasses),
      gravity(gravity),
      massCart(massCart),
      tau(tau),
      thetaThresholdRadians(thetaThresholdRadians),
      xThreshold(xThreshold),
      doneReward(doneReward),
      maxSteps(maxSteps),
      stepsPerformed(0)
  {
    if (poleNum != poleLengths.n_elem)
    {
      Log::Fatal << "The number of lengths should be the same as the number of"
          "poles." << std::endl;
    }
    if (poleNum != poleMasses.n_elem)
    {
      Log::Fatal << "The number of masses should be the same as the number of"
          "poles." << std::endl;
    }
  }

  /**
   * Dynamics of Continuous Multiple Pole Cart instance. Get reward and next state 
   * based on current state and current action.
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

    // Calculate acceleration.
    double totalForce = action.action;
    double totalMass = massCart;
    for (size_t i = 0; i < poleNum; i++)
    {
      double poleOmega = state.AngularVelocity(i + 1);
      double sinTheta = sin(state.Angle(i + 1));
      totalForce += (poleMasses[i] * poleLengths[i] * poleOmega * poleOmega *
          sinTheta) + 0.75 * poleMasses[i] * gravity * sin(2 * state.Angle(i +
          1)) / 2;
      totalMass += poleMasses[i] * (0.25 + 0.75 * sinTheta * sinTheta);
    }
    double xAcc = totalForce / totalMass;

    // Update states of the poles.
    for (size_t i = 1; i <= poleNum; i++)
    {
      double sinTheta = sin(state.Angle(i));
      double cosTheta = cos(state.Angle(i));
      nextState.Angle(i) = state.Angle(i) + tau * state.AngularVelocity(i);
      nextState.AngularVelocity(i) = state.AngularVelocity(i) - tau * 0.75 *
          (xAcc * cosTheta + gravity * sinTheta) / poleLengths[i - 1];
    }

    // Update state of the cart.
    nextState.Position() = state.Position() + tau * state.Velocity();
    nextState.Velocity() = state.Velocity() + tau * xAcc;

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
   * Dynamics of Continuous Multiple Pole Cart. Get reward based on current
   * state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, it's always 1.0.
   */
  double Sample(const State& state, const Action& action)
  {
    State nextState(poleNum);
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
    return State((arma::randu<arma::mat>(2, poleNum + 1) - 0.5) / 10.0);
  }

  /**
   * This function checks if the cart has reached the terminal state.
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
    for (size_t i = 1; i <= poleNum; i++)
    {
      if (std::abs(state.Angle(i)) > thetaThresholdRadians)
      {
        Log::Info << "Episode terminated due to pole falling";
        return true;
      }
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
  //! Locally-stored number of poles.
  size_t poleNum;

  //! Locally-stored length of poles.
  arma::vec poleLengths;

  //! Locally-stored mass of the pole.
  arma::vec poleMasses;

  //! Locally-stored gravity.
  double gravity;

  //! Locally-stored mass of the cart.
  double massCart;

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

} // namespace rl
} // namespace mlpack

#endif
