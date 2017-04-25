/**
 * @file cart_pole.hpp
 * @author Shangtong Zhang
 *
 * This file is an implementation of Cart Pole task
 * https://gym.openai.com/envs/CartPole-v0
 *
 * TODO: refactor to OpenAI interface
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_CART_POLE_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_CART_POLE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace rl {

namespace cart_pole_details {
// Some constants of Cart Pole task
constexpr double gravity = 9.8;
constexpr double massCart = 1.0;
constexpr double massPole = 0.1;
constexpr double totalMass = massCart + massPole;
constexpr double length = 0.5;
constexpr double poleMassLength = massPole * length;
constexpr double forceMag = 10.0;
constexpr double tau = 0.02;
constexpr double thetaThresholdRadians = 12 * 2 * 3.1416 / 360;
constexpr double xThreshold = 2.4;
}

/**
 * Implementation of Cart Pole task
 */
class CartPole
{
 public:

  /**
   * Implementation of state of Cart Pole
   * Each state is a tuple of (position, velocity, angle, angular velocity)
   */
  class State
  {
   public:
    //! Construct a state instance
    State() : data(4) { }

    //! Construct a state instance from given data
    State(arma::colvec data) : data(data) { }

    //! Get position
    double X() const
    {
      return data[0];
    }

    //! Modify position
    double& X()
    {
      return data[0];
    }

    //! Get velocity
    double XDot() const
    {
      return data[1];
    }

    //! Modify velocity
    double& XDot()
    {
      return data[1];
    }

    //! Get angle
    double Theta() const
    {
      return data[2];
    }

    //! Modify angle
    double& Theta()
    {
      return data[2];
    }

    //! Get angular velocity
    double ThetaDot() const
    {
      return data[3];
    }

    //! Modify angular velocity
    double& ThetaDot()
    {
      return data[3];
    }

    //! Encode the state to a column vector
    const arma::colvec& Encode() const
    {
      return data;
    }

    //! Whether current state is terminal state
    bool IsTerminal() const
    {
      using namespace cart_pole_details;
      return std::abs(X()) > xThreshold ||
             std::abs(Theta()) > thetaThresholdRadians;
    }

   private:
    //! Locally-stored (position, velocity, angle, angular velocity)
    arma::colvec data;
  };

  /**
   * Implementation of action of Cart Pole
   */
  class Action
  {
   public:
    enum Actions
    {
      backward,
      forward
    };

    //! # of actions
    static constexpr size_t count = 2;
  };

  /**
   * Dynamics of Cart Pole
   * Get next state and next action based on current state and current action
   * @param state Current state
   * @param action Current action
   * @param nextState Next state
   * @param reward Reward is always 1
   */
  void Sample(const State& state, const Action::Actions& action,
              State& nextState, double& reward)
  {
    using namespace cart_pole_details;
    double force = action ? forceMag : -forceMag;
    double cosTheta = std::cos(state.Theta());
    double sinTheta = std::sin(state.Theta());
    double temp = (force + poleMassLength * state.ThetaDot() * state.ThetaDot() * sinTheta) / totalMass;
    double thetaAcc = (gravity * sinTheta - cosTheta * temp) /
            (length * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass));
    double xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;
    nextState.X() = state.X() + tau * state.XDot();
    nextState.XDot() = state.XDot() + tau * xAcc;
    nextState.Theta() = state.Theta() + tau * state.ThetaDot();
    nextState.ThetaDot() = state.ThetaDot() + tau * thetaAcc;

    reward = 1.0;
  }

  /**
   * Initial state representation is randomly generated within [-0.05, 0.05]
   * @return Initial state for each episode
   */
  State InitialSample()
  {
    return State((arma::randu<arma::colvec>(4) - 0.5) / 10.0);
  }

};

} // namespace rl
} // namespace mlpack

#endif