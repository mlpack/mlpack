/**
 * @file pendulum.hpp
 * @author Shashank Shekhar
 *
 * This file is an implementation of Pendulum task:
 * https://gym.openai.com/envs/Pendulum-v0/
 *
 * TODO: provide an option to use dynamics directly from OpenAI gym.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_PENDULUM_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_PENDULUM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

/**
 * Implementation of Pendulum task. The inverted pendulum swingup problem is a
 * classic problem in the control literature. In this version of the problem,
 * the pendulum starts in a random position, and the goal is to swing it up so
 * it stays upright
 */
class Pendulum
{
 public:
  /**
   * Implementation of state of Pendulum. Each state is a
   * (theta, angular velocity) vector.
   */
  class State
  {
   public:
    /**
     * Construct a state instance.
     */
    State() : data(dimension, arma::fill::zeros)
    { /* Nothing to do here. */ }

    /**
     * Construct a state based on the given data.
     *
     * @param data Data for the theta and angular velocity.
     */
    State(const arma::colvec& data): data(data)
    { /* Nothing to do here. */ }

    //! Modify the internal representation of the state.
    arma::colvec& Data() { return data; }

    //! Get the theta.
    double Theta() const { return data[0]; }
    //! Modify the value of theta.
    double& Theta() { return data[0]; }

    //! Get the angular velocity.
    double AngularVelocity() const { return data[1]; }
    //! Modify the value of angular velocity.
    double& AngularVelocity() { return data[1]; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 2;

   private:
    //! Locally-stored (theta, angular velocity) vector
    arma::colvec data;
  };

  /**
   * Implementation of action of Pendulum.
   * In Pendulum, the action represents the torque to be applied.
   * This value is bounded in range -2.0 to 2.0 by default.
   */
  struct Action
  {
    double action[1];
    // Storing degree of freedom
    const int size = 1;
  };

  /**
   * Construct a Pendulum instance using the given values.
   *
   * @param maxAngularVelocity Maximum angular velocity.
   * @param maxTorque Maximum torque.
   * @param dt The differential value.
   */
  Pendulum(const double maxAngularVelocity = 8,
           const double maxTorque = 2.0,
           const double dt = 0.05) :
      maxAngularVelocity(maxAngularVelocity),
      maxTorque(maxTorque),
      dt(dt)
  { /* Nothing to do here */ }

  /**
   * Dynamics of Pendulum. Get reward and next state based 
   * on current state and current action.
   *
   * @param state The current state.
   * @param action The current action.
   * @param nextState The next state.
   * @return reward, The reward for taking the action taken for current state.
   */
  double Sample(const State& state,
                const Action& action,
                State& nextState) const
  {
    // Get current state.
    double theta = state.Theta();
    double angularVelocity = state.AngularVelocity();

    // Define constants which specify our pendulum.
    const double gravity = 10.0;
    const double mass = 1.0;
    const double length = 1.0;

    // Get action and clip the values between max and min limits.
    double torque = std::min(
        std::max(action.action[0], -maxTorque), maxTorque);

    // Calculate costs of taking this action in the current state.
    double costs = std::pow(AngleNormalize(theta), 2) + 0.1 *
        std::pow(angularVelocity, 2) + 0.001 * std::pow(torque, 2);

    // Calculate new state values and assign to the next state.
    double newAngularVelocity = angularVelocity + (-3.0 * gravity / (2 *
        length) * std::sin(theta + M_PI) + 3.0 / std::pow(mass * length, 2) *
        torque) * dt;
    nextState.AngularVelocity() = std::min(std::max(newAngularVelocity,
        -maxAngularVelocity), maxAngularVelocity);
    nextState.Theta() = theta + newAngularVelocity * dt;

    // Return the reward of taking the action in current state.
    // The reward is simply the negative of cost incurred for the action.
    return -costs;
  }

  /**
   * Dynamics of Pendulum. Get reward based on current state and current action
   *
   * @param state The current state.
   * @param action The current action.
   * @return reward, The reward.
   */
  double Sample(const State& state, const Action& action) const
  {
    State nextState;
    return Sample(state, action, nextState);
  }

  /**
   * Initial theta is randomly generated within [-pi, pi].
   * Initial angular velocity is randomly generated within [-1, 1].
   *
   * @return Initial state for each episode.
   */
  State InitialSample() const
  {
    State state;
    state.Theta() = math::Random(-M_PI, M_PI);
    state.AngularVelocity() = math::Random(-1.0, 1.0);
    return state;
  }

  /**
   * This function calculates the normalized anlge for a particular theta.
   *
   * @param theta The un-normalized angle.
   */
  double AngleNormalize(double theta) const
  {
    // Scale angle within [-pi, pi).
    return double(fmod(theta + M_PI, 2 * M_PI) - M_PI);
  }

 private:
  //! Locally-stored maximum legal angular velocity.
  double maxAngularVelocity;

  //! Locally-stored maximum legal torque.
  double maxTorque;

  //! Locally-stored dt.
  double dt;
};

} // namespace rl
} // namespace mlpack

#endif
