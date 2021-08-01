
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_double_pole_cart.hpp:

Program Listing for File double_pole_cart.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_double_pole_cart.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_DOUBLE_POLE_CART_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_DOUBLE_POLE_CART_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace rl {
   
   class DoublePoleCart
   {
    public:
     class State
     {
      public:
       State() : data(dimension)
       { /* Nothing to do here. */ }
   
       State(const arma::colvec& data) : data(data)
       { /* Nothing to do here */ }
   
       arma::colvec Data() const { return data; }
       arma::colvec& Data() { return data; }
   
       double Position() const { return data[0]; }
       double& Position() { return data[0]; }
   
       double Velocity() const { return data[1]; }
       double& Velocity() { return data[1]; }
   
       double Angle(const size_t i) const { return data[2 * i]; }
       double& Angle(const size_t i) { return data[2 * i]; }
   
       double AngularVelocity(const size_t i) const { return data[2 * i + 1]; }
       double& AngularVelocity(const size_t i) { return data[2 * i + 1]; }
   
       const arma::colvec& Encode() const { return data; }
   
       static constexpr size_t dimension = 6;
   
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
   
     DoublePoleCart(const size_t maxSteps = 0,
                    const double m1 = 0.1,
                    const double m2 = 0.01,
                    const double l1 = 0.5,
                    const double l2 = 0.05,
                    const double gravity = 9.8,
                    const double massCart = 1.0,
                    const double forceMag = 10.0,
                    const double tau = 0.02,
                    const double thetaThresholdRadians = 36 * 2 * 3.1416 / 360,
                    const double xThreshold = 2.4,
                    const double doneReward = 0.0) :
         maxSteps(maxSteps),
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
         stepsPerformed(0)
     { /* Nothing to do here */ }
   
     double Sample(const State& state,
                   const Action& action,
                   State& nextState)
     {
       // Update the number of steps performed.
       stepsPerformed++;
   
       arma::vec dydx(6, arma::fill::zeros);
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
   
       return 1.0;
     }
   
     void Dsdt(const State& state,
               const Action& action,
               arma::vec& dydx)
     {
       double totalForce = action.action ? forceMag : -forceMag;
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
   
     double Sample(const State& state, const Action& action)
     {
       State nextState;
       return Sample(state, action, nextState);
     }
   
     State InitialSample()
     {
       stepsPerformed = 0;
       return State((arma::randu<arma::vec>(6) - 0.5) / 10.0);
     }
   
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
   
     size_t StepsPerformed() const { return stepsPerformed; }
   
     size_t MaxSteps() const { return maxSteps; }
     size_t& MaxSteps() { return maxSteps; }
   
    private:
     size_t maxSteps;
   
     double m1;
   
     double m2;
   
     double l1;
   
     double l2;
   
     double gravity;
   
     double massCart;
   
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
