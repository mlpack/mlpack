
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_cart_pole.hpp:

Program Listing for File cart_pole.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_cart_pole.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/cart_pole.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_CART_POLE_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_CART_POLE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace rl {
   
   class CartPole
   {
    public:
     class State
     {
      public:
       State() : data(dimension)
       { /* Nothing to do here. */ }
   
       State(const arma::colvec& data) : data(data)
       { /* Nothing to do here */ }
   
       arma::colvec& Data() { return data; }
   
       double Position() const { return data[0]; }
       double& Position() { return data[0]; }
   
       double Velocity() const { return data[1]; }
       double& Velocity() { return data[1]; }
   
       double Angle() const { return data[2]; }
       double& Angle() { return data[2]; }
   
       double AngularVelocity() const { return data[3]; }
       double& AngularVelocity() { return data[3]; }
   
       const arma::colvec& Encode() const { return data; }
   
       static constexpr size_t dimension = 4;
   
      private:
       arma::colvec data;
     };
   
     class Action
     {
      public:
       enum actions
       {
         backward,
         forward
       };
       // To store the action.
       Action::actions action;
   
       // Track the size of the action space.
       static const size_t size = 2;
     };
   
     CartPole(const size_t maxSteps = 200,
              const double gravity = 9.8,
              const double massCart = 1.0,
              const double massPole = 0.1,
              const double length = 0.5,
              const double forceMag = 10.0,
              const double tau = 0.02,
              const double thetaThresholdRadians = 12 * 2 * 3.1416 / 360,
              const double xThreshold = 2.4,
              const double doneReward = 1.0) :
         maxSteps(maxSteps),
         gravity(gravity),
         massCart(massCart),
         massPole(massPole),
         totalMass(massCart + massPole),
         length(length),
         poleMassLength(massPole * length),
         forceMag(forceMag),
         tau(tau),
         thetaThresholdRadians(thetaThresholdRadians),
         xThreshold(xThreshold),
         doneReward(doneReward),
         stepsPerformed(0)
     { /* Nothing to do here */ }
   
     double Sample(const State& state,
                   const Action& action,
                   State& nextState)
     {
       // Update the number of steps performed.
       stepsPerformed++;
   
       // Calculate acceleration.
       double force = action.action ? forceMag : -forceMag;
       double cosTheta = std::cos(state.Angle());
       double sinTheta = std::sin(state.Angle());
       double temp = (force + poleMassLength * state.AngularVelocity() *
           state.AngularVelocity() * sinTheta) / totalMass;
       double thetaAcc = (gravity * sinTheta - cosTheta * temp) /
           (length * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass));
       double xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;
   
       // Update states.
       nextState.Position() = state.Position() + tau * state.Velocity();
       nextState.Velocity() = state.Velocity() + tau * xAcc;
       nextState.Angle() = state.Angle() + tau * state.AngularVelocity();
       nextState.AngularVelocity() = state.AngularVelocity() + tau * thetaAcc;
   
       // Check if the episode has terminated.
       bool done = IsTerminal(nextState);
   
       // Do not reward agent if it failed.
       if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
         return doneReward;
   
       return 1.0;
     }
   
     double Sample(const State& state, const Action& action)
     {
       State nextState;
       return Sample(state, action, nextState);
     }
   
     State InitialSample()
     {
       stepsPerformed = 0;
       return State((arma::randu<arma::colvec>(4) - 0.5) / 10.0);
     }
   
     bool IsTerminal(const State& state) const
     {
       if (maxSteps != 0 && stepsPerformed >= maxSteps)
       {
         Log::Info << "Episode terminated due to the maximum number of steps"
             "being taken.";
         return true;
       }
       else if (std::abs(state.Position()) > xThreshold ||
           std::abs(state.Angle()) > thetaThresholdRadians)
       {
         Log::Info << "Episode terminated due to agent failing.";
         return true;
       }
       return false;
     }
   
     size_t StepsPerformed() const { return stepsPerformed; }
   
     size_t MaxSteps() const { return maxSteps; }
     size_t& MaxSteps() { return maxSteps; }
   
    private:
     size_t maxSteps;
   
     double gravity;
   
     double massCart;
   
     double massPole;
   
     double totalMass;
   
     double length;
   
     double poleMassLength;
   
     double forceMag;
   
     double tau;
   
     double thetaThresholdRadians;
   
     double xThreshold;
   
     double doneReward;
   
     size_t stepsPerformed;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
