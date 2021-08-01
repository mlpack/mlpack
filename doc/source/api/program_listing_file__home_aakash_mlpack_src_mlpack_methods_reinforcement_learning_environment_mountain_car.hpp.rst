
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_mountain_car.hpp:

Program Listing for File mountain_car.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_mountain_car.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/mountain_car.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_MOUNTAIN_CAR_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_MOUNTAIN_CAR_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/clamp.hpp>
   
   namespace mlpack {
   namespace rl {
   
   class MountainCar
   {
    public:
     class State
     {
      public:
       State(): data(dimension, arma::fill::zeros)
       { /* Nothing to do here. */ }
   
       State(const arma::colvec& data): data(data)
       { /* Nothing to do here. */ }
   
       arma::colvec& Data() { return data; }
   
       double Velocity() const { return data[0]; }
       double& Velocity() { return data[0]; }
   
       double Position() const { return data[1]; }
       double& Position() { return data[1]; }
   
       const arma::colvec& Encode() const { return data; }
   
       static constexpr size_t dimension = 2;
   
      private:
       arma::colvec data;
     };
   
     class Action
     {
      public:
       enum actions
       {
         backward,
         stop,
         forward
       };
       // To store the action.
       Action::actions action;
   
       // Track the size of the action space.
       static const size_t size = 3;
     };
   
     MountainCar(const size_t maxSteps = 200,
                 const double positionMin = -1.2,
                 const double positionMax = 0.6,
                 const double positionGoal = 0.5,
                 const double velocityMin = -0.07,
                 const double velocityMax = 0.07,
                 const double doneReward = 0) :
         maxSteps(maxSteps),
         positionMin(positionMin),
         positionMax(positionMax),
         positionGoal(positionGoal),
         velocityMin(velocityMin),
         velocityMax(velocityMax),
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
       int direction = action.action - 1;
       nextState.Velocity() = state.Velocity() + 0.001 * direction - 0.0025 *
           std::cos(3 * state.Position());
       nextState.Velocity() = math::ClampRange(nextState.Velocity(),
           velocityMin, velocityMax);
   
       // Update states.
       nextState.Position() = state.Position() + nextState.Velocity();
       nextState.Position() = math::ClampRange(nextState.Position(),
           positionMin, positionMax);
   
       if (nextState.Position() == positionMin && nextState.Velocity() < 0)
         nextState.Velocity() = 0.0;
   
       // Check if the episode has terminated.
       bool done = IsTerminal(nextState);
   
       // Do not reward the agent if time ran out.
       if (done && maxSteps != 0 && stepsPerformed >= maxSteps)
         return 0;
       else if (done)
         return doneReward;
   
       return -1;
     }
   
     double Sample(const State& state, const Action& action)
     {
       State nextState;
       return Sample(state, action, nextState);
     }
   
     State InitialSample()
     {
       State state;
       stepsPerformed = 0;
       state.Velocity() = 0.0;
       state.Position() = arma::as_scalar(arma::randu(1)) * 0.2 - 0.6;
       return state;
     }
   
     bool IsTerminal(const State& state) const
     {
       if (maxSteps != 0 && stepsPerformed >= maxSteps)
       {
         Log::Info << "Episode terminated due to the maximum number of steps"
             "being taken.";
         return true;
       }
       else if (state.Position() >= positionGoal)
       {
         Log::Info << "Episode terminated due to agent succeeding.";
         return true;
       }
       return false;
     }
   
     size_t StepsPerformed() const { return stepsPerformed; }
   
     size_t MaxSteps() const { return maxSteps; }
     size_t& MaxSteps() { return maxSteps; }
   
    private:
     size_t maxSteps;
   
     double positionMin;
   
     double positionMax;
   
     double positionGoal;
   
     double velocityMin;
   
     double velocityMax;
   
     double doneReward;
   
     size_t stepsPerformed;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
