/**
 * @file pong.hpp
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
#define PIE 3.14159265358979323846

#include <math.h>
#include <mlpack/prereqs.hpp>

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

      //! Get sin() value of theta1
      double Theta1() const {return data[0];}
      //! Modify sin() value of theta1
      double& Theta1() {return data[0];}

      // double theta1COS() const {return data[1];}
      // //! Modify cos() value of theta1
      // double& theta1COS() {return data[1];}

      double Theta2() const {return data[1];}
      //! Modify cos() value of theta1
      double& Theta2() {return data[1];}

      // double theta2COS() const {return data[3];}
      // //! Modify cos() value of theta1
      // double& theta2COS() {return data[3];}

      double AngularVelocity1() const { return data[3]; }
      //! Modify the angular velocity.
      double& AngularVelocity1() { return data[3]; }

      double AngularVelocity2() const { return data[4]; }
      //! Modify the angular velocity.
      double& AngularVelocity2() { return data[4]; }

      //! Encode the state to a column vector.
      const arma::colvec& Encode() const { return data; }

      //! Dimension of the encoded state.
      static constexpr size_t dimension = 4;

     private :
      arma::colvec data;
   }; // state 
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
   * @param link_length1 length of link 1.
   * @param link_length2 length of link 2.
   * @param link_mass1 mass of link 1.
   * @param link_mass2 mass of link 2.
   * @param link_com1 position of the center of mass of link 1.
   * @param link_com2 position of the center of mass of link 2.
   * @param link_moi moments of inertia for both link.
   * @param max_vel1 max angular velocity of link1.
   * @param max_vel2 max angular velocity of link2.
   */
   Acrobat(const double gravity = 9.81,
   	       const double link_length1 = 1.0,
           const double link_length2 = 1.0,
           const double link_mass1 = 1.0,
           const double link_mass2 = 1.0,
           const double link_com1 = 0.5,
           const double link_com2 = 0.5,
           const double link_moi = 1.0,
           const double max_vel1 = 4*PIE,
           const double max_vel2 = 9*PIE,
           const double dt = 0.2) :
      gravity(gravity),
      link_length1(link_length1),
      link_length2(link_length2),
      link_mass1(link_mass1),
      link_mass2(link_mass2),
      link_com1(link_com1),
      link_com2(link_com2),
      link_moi(link_moi),
      max_vel1(max_vel1),
      max_vel2(max_vel2),
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
                 State& nextState) const
          {
          	double torque = action - 1;
            // nextState = rk4(state,action);
            return -1;
          }; // Sample
   arma::colvec dsdt(const State& state,
   	           const int torque)
   	      {
            double m1 = link_mass1;
            double m2 = link_mass2;
            double l1 = link_length1;
            double l2 = link_length2;
            double lc1 = link_com1;
            double lc2 = link_com2;
            double I1 = link_moi;
            double I2 = link_moi;
            double g = gravity;
            int a = torque;
            double theta1 = state.Theta1();
            double theta2 = state.Theta2();
            double dtheta1 = state.AngularVelocity1();
            double dtheta2 = state.AngularVelocity2();
            double d1 = m1 * pow(lc1,2) + m2 * 
                       (pow(l1,2) + pow(lc2,2) + 2 * l1 * lc2 * cos(theta2)) 
                        + I1 + I2 ;
            double d2 = m2 * (pow(lc2,2) + l1 * lc2 * cos(theta2)) + I2;
            double phi2 = m2 * lc2 * g * cos(theta1 + theta2 - PIE / 2.);
            
            double phi1 = - m2 * l1 * lc2 * pow(dtheta2,2) * sin(theta2)
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
               + (m1 * lc1 + m2 * l1) * g * cos(theta1 - PIE / 2)
               + phi2;
            
            double ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * pow(dtheta1,2) * 
            	sin(theta2) - phi2) / (m2 * pow(lc2,2) + I2 - pow(d2,2) / d1);

            double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
            arma::colvec returnValues = {dtheta1, dtheta2, ddtheta1, ddtheta2};
            return returnValues;
   	      };
   	/**
   	* @param value scalar value to wrap
   	* @param min minimum range of wrap
   	* @param max maximum range of wrap
   	*/      
   double wrap(double value,
               double min,
               double max)
          {
          	double diff = max - min;
          	if (value>max) value = value - diff;
          	else if (value<min) value = value + diff;
          	return value;
          }; 
   double  bound()
          { };
   // double 
 private:
   //! Locally-stored gravity.
   double gravity;

   //! Locally-stored length of link 1.
   double link_length1;

   //! Locally-stored length of link 2.
   double link_length2;

   //! Locally-stored mass of link 1.
   double link_mass1;
   
   //! Locally-stored mass of link 2.
   double link_mass2;

   //! Locally-stored position of link 1.
   double link_com1;

   //! Locally-stored position of link 2.
   double link_com2;

   //! Locally-stored moment of intertia value.
   double link_moi;

   //! Locally-stored max angular velocity of link1.
   double max_vel1;

   //! Locally-stored max angular velocity of link2.
   double max_vel2;

   //! Locally-stored dt for RK4 method
   double dt;      

}; // class Acrobat
} // namespace rl
} // namespace mlpack