
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_pendulum.hpp:

Program Listing for File pendulum.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_pendulum.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/pendulum.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_PENDULUM_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_PENDULUM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/clamp.hpp>
   
   namespace mlpack {
   namespace rl {
   
   class Pendulum
   {
    public:
     class State
     {
      public:
       State() : theta(0), data(dimension, arma::fill::zeros)
       { /* Nothing to do here. */ }
   
       State(const arma::colvec& data): theta(0), data(data)
       { /* Nothing to do here. */ }
   
       arma::colvec& Data() { return data; }
   
       double Theta() const { return theta; }
       double& Theta() { return theta; }
   
       double AngularVelocity() const { return data[2]; }
       double& AngularVelocity() { return data[2]; }
   
       const arma::colvec& Encode() { return data; }
   
       void SetState()
       {
         data[0] = std::sin(theta);
         data[1] = std::cos(theta);
       }
   
       static constexpr size_t dimension = 3;
   
      private:
       double theta;
   
       arma::colvec data;
     };
   
     class Action
     {
      public:
       Action() : action(1)
       { /* Nothing to do here */ }
       std::vector<double> action;
       // Storing degree of freedom.
       static const size_t size = 1;
     };
   
     Pendulum(const size_t maxSteps = 200,
              const double maxAngularVelocity = 8,
              const double maxTorque = 2.0,
              const double dt = 0.05,
              const double doneReward = 0.0) :
         maxSteps(maxSteps),
         maxAngularVelocity(maxAngularVelocity),
         maxTorque(maxTorque),
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
   
       // Get current state.
       double theta = state.Theta();
       double angularVelocity = state.AngularVelocity();
   
       // Define constants which specify our pendulum.
       const double gravity = 10.0;
       const double mass = 1.0;
       const double length = 1.0;
   
       // Get action and clip the values between max and min limits.
       double torque = math::ClampRange(action.action[0], -maxTorque, maxTorque);
   
       // Calculate costs of taking this action in the current state.
       double costs = std::pow(AngleNormalize(theta), 2) + 0.1 *
           std::pow(angularVelocity, 2) + 0.001 * std::pow(torque, 2);
   
       // Calculate new state values and assign to the next state.
       double newAngularVelocity = angularVelocity + (-3.0 * gravity / (2 *
           length) * std::sin(theta + M_PI) + 3.0 / (mass * std::pow(length, 2)) *
           torque) * dt;
       nextState.Theta() = theta + newAngularVelocity * dt;
       nextState.AngularVelocity() = math::ClampRange(newAngularVelocity,
           -maxAngularVelocity, maxAngularVelocity);
   
       nextState.SetState();
   
       // Return the reward of taking the action in current state.
       // The reward is simply the negative of cost incurred for the action.
       return -costs;
     }
   
     double Sample(const State& state, const Action& action)
     {
       State nextState;
       return Sample(state, action, nextState);
     }
   
     State InitialSample()
     {
       State state;
       state.Theta() = math::Random(-M_PI, M_PI);
       state.AngularVelocity() = math::Random(-1.0, 1.0);
       stepsPerformed = 0;
       state.SetState();
       return state;
     }
   
     double AngleNormalize(double theta) const
     {
       // Scale angle within [-pi, pi).
       double x = fmod(theta + M_PI, 2 * M_PI);
       if (x < 0)
         x += 2 * M_PI;
       return x - M_PI;
     }
   
     bool IsTerminal(const State& /* state */) const
     {
       if (maxSteps != 0 && stepsPerformed >= maxSteps)
       {
         Log::Info << "Episode terminated due to the maximum number of steps"
             "being taken.";
         return true;
       }
       return false;
     }
   
     size_t StepsPerformed() const { return stepsPerformed; }
   
     size_t MaxSteps() const { return maxSteps; }
     size_t& MaxSteps() { return maxSteps; }
   
    private:
     size_t maxSteps;
   
     double maxAngularVelocity;
   
     double maxTorque;
   
     double dt;
   
     double doneReward;
   
     size_t stepsPerformed;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
