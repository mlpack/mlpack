/**
 * @file acrobat.hpp
 * @author Rohan Raj
 *
 * This file is an implementation of Acrobat task:
 * https://gym.openai.com/envs/Acrobot-v1/
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ENVIRONMENT_ACROBAT_HPP
#define MLPACK_METHODS_RL_ENVIRONMENT_ACROBAT_HPP

#include <mlpack/core.hpp>

namespace mlpack{
namespace rl{
/*
 * Implementation of Acrobat game
 * Acrobot is a 2-link pendulum with only the second joint actuated
 * Intitially, both links point downwards. The goal is to swing the
 * end-effector at a height at least the length of one link above the base.
 * Both links can swing freely and can pass by each other, i.e., they don't
 * collide when they have the same angle.
 */
class Acrobat
{
 public:
   /*
    * Implementation of Acrobat State
    */    
  class State
  {
   public :
    /**
    * Construct a state instance.
    */
    State(): data(dimension)
    { /* nothing to do here */ }
    /**
    * Construct a state instance from given data.
    *
    * @param data Data for the acrobat.
    * Consists of sin() and cos() for two rotational joint and angular velocity.
    * 
    */
    State(const arma::colvec& data): data(data)
    { /* nothing to do here */ }

    //! Modify the state representation
    arma::colvec& Data() {return data;}

    //! Get value of theta1
    double Theta1() const {return data[0];}
    //! Modify value of theta1
    double& Theta1() {return data[0];}

    //! Get value of theta2
    double Theta2() const {return data[1];}
    //! Modify value of theta2
    double& Theta2() {return data[1];}

    //! Get value of Angular velocity 1
    double AngularVelocity1() const { return data[2]; }
    //! Modify the angular velocity 1.
    double& AngularVelocity1() { return data[2]; }

    //! Get value of Angular velocity 2
    double AngularVelocity2() const { return data[3]; }
    //! Modify the angular velocity 2.
    double& AngularVelocity2() { return data[3]; }

    //! Encode the state to a column vector.
    const arma::colvec& Encode() const { return data; }

    //! Dimension of the encoded state.
    static constexpr size_t dimension = 4;

   private :
    //! Locally-Stored (theta1, theta2, angular velocity 1, angular velocity2)
    arma::colvec data;
  };
  /* 
   * Implementation of action for cartpole
   */
  enum Action
  {
    negativeTorque,
    zeroTorque,
    positiveTorque,
    // Track the size of the action space.
    size
  };

   /**
   * Construct a Cart Pole instance using the given constants.
   *
   * @param gravity gravity
   * @param linkLength1 length of link 1.
   * @param linkLength2 length of link 2.
   * @param linkMass1 mass of link 1.
   * @param linkMass2 mass of link 2.
   * @param linkCom1 position of the center of mass of link 1.
   * @param linkCom2 position of the center of mass of link 2.
   * @param linkMoi moments of inertia for both link.
   * @param maxVel1 max angular velocity of link1.
   * @param maxVel2 max angular velocity of link2.
   */
  Acrobat(const double gravity = 9.81,
          const double linkLength1 = 1.0,
          const double linkLength2 = 1.0,
          const double linkMass1 = 1.0,
          const double linkMass2 = 1.0,
          const double linkCom1 = 0.5,
          const double linkCom2 = 0.5,
          const double linkMoi = 1.0,
          const double maxVel1 = 4*M_PI,
          const double maxVel2 = 9*M_PI,
          const double dt = 0.2) :
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
      dt(dt)
  { /* Nothing to do here */ }
  /**
    * Dynamics of the Acrobat System.
    * To get reward and next state based on current
    * state and current action .
    * 
    * @param state The current State
    * @param action The action taken
    * @param nextState The next state
    * return reward -1
    * Always return -1 reward
    */
  double Sample(const State& state,
                 const Action& action,
                 State& nextState)
  {
    Rk4(state, action, nextState);
    nextState.Theta1() = Wrap(nextState.Theta1(), -M_PI, M_PI);

    nextState.Theta2() = Wrap(nextState.Theta2(), -M_PI, M_PI);
    nextState.AngularVelocity1() = Bound(nextState.AngularVelocity1(),
                                             -maxVel1 , maxVel1);
    nextState.AngularVelocity2() = Bound(nextState.AngularVelocity2(),
                                             -maxVel2 , maxVel2);
    return -1;
  };
  double Sample(const State& state, const Action& action)
  {
    State nextState;
    return Sample(state, action, nextState);
  }
  State InitialSample() const
  {
    return State((arma::randu<arma::colvec>(4) - 0.5) / 5.0);
  }
  /**
   *
   * @param state The current State
   */        
  bool IsTerminal(const State& state) const
  {
    return bool (-cos(state.Theta1())-cos(state.Theta1()
      + state.Theta2()) > 1.0);
  }
  /**
   * @param state Current State
   * @param torque Torque Applied 
   */
  arma::colvec Dsdt(arma::colvec state,
               const double torque)
  {
    double m1 = linkMass1;
    double m2 = linkMass2;
    double l1 = linkLength1;
    double l2 = linkLength2;
    double lc1 = linkCom1;
    double lc2 = linkCom2;
    double I1 = linkMoi;
    double I2 = linkMoi;
    double g = gravity;
    double a = torque;
    double theta1 = state[0];
    double theta2 = state[1];
    double dtheta1 = state[2];
    double dtheta2 = state[3];
    double d1 = m1 * pow(lc1, 2) + m2 *
              (pow(l1, 2) + pow(lc2, 2) + 2 * l1 * lc2 * cos(theta2))
               + I1 + I2;
    double d2 = m2 * (pow(lc2, 2) + l1 * lc2 * cos(theta2)) + I2;
    double phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI / 2.);

    double phi1 = - m2 * l1 * lc2 * pow(dtheta2, 2) * sin(theta2)
      - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
      + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI / 2)
      + phi2;

    double ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * pow(dtheta1, 2) *
     sin(theta2) - phi2) / (m2 * pow(lc2, 2) + I2 - pow(d2, 2) / d1);

    double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
    arma::colvec returnValues = {dtheta1, dtheta2, ddtheta1, ddtheta2};
    return returnValues;
  };
  /**
   * @param value scalar value to wrap
   * @param minimum minimum range of wrap
   * @param maximum maximum range of wrap
   */      
  double Wrap(double value,
              double minimum,
              double maximum)
  {
    double diff = maximum - minimum;
    if (value > maximum) value = value - diff;
    else if (value < minimum) value = value + diff;
    return value;
  };
  /**
    * @param value scalar value to bound
    * @param minimum minimum range of bound
    * @param maximum maximum range of bound
    */     
  double  Bound(double value,
                double minimum,
                double maximum)
  {
    return std::min(std::max(value, minimum), maximum);
  };

  /**
   * @param Action action taken
   * 0 : negative torque
   * 1 : zero torque
   * 2 : positive torque
   */
  double Torque(const Action& Action)
  {
    // Add noise to the Torque
    /*
    * Torque is action number - 1.
    * {0,1,2} -> {-1,0,1} 
    */
    double torque = double(Action - 1) + Random(-0.1, 0.1);
    return torque;
  }

  /**
    * @param state_ Current State
    * @param Action Action Taken
    * @param nextState nextState 
    */
  void Rk4(const State& state_,
           const Action& Action,
           State& nextState)
  {
  /*
   * Torque is action number - 1.
   * {0,1,2} -> {-1,0,1} 
   */
    double torque = Torque(Action);
    arma::colvec state = {state_.Theta1(), state_.Theta2(),
                          state_.AngularVelocity1(),
                          state_.AngularVelocity2()};
    arma::colvec k1 = Dsdt(state, torque);
    arma::colvec k2 = Dsdt(state + dt*k1/2, torque);
    arma::colvec k3 = Dsdt(state + dt*k2/2, torque);
    arma::colvec k4 = Dsdt(state + dt*k3, torque);
    arma::colvec nextstate = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6;
    nextState.Theta1() = nextstate[0];
    nextState.Theta2() = nextstate[1];
    nextState.AngularVelocity1() = nextstate[2];
    nextState.AngularVelocity2() = nextstate[3];
  };
 private:
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

  //! Locally-stored dt for RK4 method
  double dt;
}; // class Acrobat
} // namespace rl
} // namespace mlpack

#endif
