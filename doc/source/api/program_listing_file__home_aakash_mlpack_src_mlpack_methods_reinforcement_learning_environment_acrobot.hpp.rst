
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_acrobot.hpp:

Program Listing for File acrobot.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_acrobot.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/acrobot.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_ACROBOT_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_ACROBOT_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack{
   namespace rl{
   
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
       State(): data(dimension) { /* nothing to do here */ }
   
       State(const arma::colvec& data) : data(data)
       { /* nothing to do here */ }
   
       arma::colvec& Data() { return data; }
   
       double Theta1() const { return data[0]; }
       double& Theta1() { return data[0]; }
   
       double Theta2() const { return data[1]; }
       double& Theta2() { return data[1]; }
   
       double AngularVelocity1() const { return data[2]; }
       double& AngularVelocity1() { return data[2]; }
   
       double AngularVelocity2() const { return data[3]; }
       double& AngularVelocity2() { return data[3]; }
   
       const arma::colvec& Encode() const { return data; }
   
       static constexpr size_t dimension = 4;
   
      private:
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
   
       nextState.AngularVelocity1() = math::ClampRange(currentNextState[2],
           -maxVel1, maxVel1);
       nextState.AngularVelocity2() = math::ClampRange(currentNextState[3],
           -maxVel2, maxVel2);
   
       // Check if the episode has terminated.
       bool done = IsTerminal(nextState);
   
       // Do not reward the agent if time ran out.
       if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
         return 0;
       else if (done)
         return doneReward;
   
       return -1;
     };
   
     double Sample(const State& state, const Action& action)
     {
       State nextState;
       return Sample(state, action, nextState);
     }
   
     State InitialSample()
     {
       stepsPerformed = 0;
       return State((arma::randu<arma::colvec>(4) - 0.5) / 5.0);
     }
   
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
   
     double Torque(const Action& action) const
     {
       // Add noise to the Torque Torque is action number - 1. {0,1,2} -> {-1,0,1}.
       return double(action.action - 1) + mlpack::math::Random(-0.1, 0.1);
     }
   
     arma::colvec Rk4(const arma::colvec state, const double torque) const
     {
       arma::colvec k1 = Dsdt(state, torque);
       arma::colvec k2 = Dsdt(state + dt * k1 / 2, torque);
       arma::colvec k3 = Dsdt(state + dt * k2 / 2, torque);
       arma::colvec k4 = Dsdt(state + dt * k3, torque);
       arma::colvec nextState = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
   
       return nextState;
     };
   
     size_t StepsPerformed() const { return stepsPerformed; }
   
     size_t MaxSteps() const { return maxSteps; }
     size_t& MaxSteps() { return maxSteps; }
   
    private:
     size_t maxSteps;
   
     double gravity;
   
     double linkLength1;
   
     double linkLength2;
   
     double linkMass1;
   
     double linkMass2;
   
     double linkCom1;
   
     double linkCom2;
   
     double linkMoi;
   
     double maxVel1;
   
     double maxVel2;
   
     double dt;
   
     double doneReward;
   
     size_t stepsPerformed;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
