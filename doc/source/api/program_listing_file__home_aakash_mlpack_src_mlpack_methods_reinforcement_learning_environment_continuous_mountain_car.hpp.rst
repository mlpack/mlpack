
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_continuous_mountain_car.hpp:

Program Listing for File continuous_mountain_car.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_continuous_mountain_car.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/continuous_mountain_car.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_CONTINUOUS_MOUNTAIN_CAR_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_CONTINUOUS_MOUNTAIN_CAR_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/clamp.hpp>
   
   namespace mlpack {
   namespace rl {
   
   class ContinuousMountainCar
   {
    public:
     class State
     {
      public:
       State() : data(dimension, arma::fill::zeros)
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
   
     struct Action
     {
       double action[1];
       // Storing degree of freedom
       const int size = 1;
     };
   
     ContinuousMountainCar(const double positionMin = -1.2,
                           const double positionMax = 0.6,
                           const double positionGoal = 0.45,
                           const double velocityMin = -0.07,
                           const double velocityMax = 0.07,
                           const double duration = 0.0015,
                           const double doneReward = 100,
                           const size_t maxSteps = 0) :
         positionMin(positionMin),
         positionMax(positionMax),
         positionGoal(positionGoal),
         velocityMin(velocityMin),
         velocityMax(velocityMax),
         duration(duration),
         doneReward(doneReward),
         maxSteps(maxSteps),
         stepsPerformed(0)
     { /* Nothing to do here */ }
   
     double Sample(const State& state,
                   const Action& action,
                   State& nextState)
     {
       // Update the number of steps performed.
       stepsPerformed++;
   
       // Calculate acceleration.
       double force = math::ClampRange(action.action[0], -1.0, 1.0);
   
       // Update states.
       nextState.Velocity() = state.Velocity() + force * duration - 0.0025 *
           std::cos(3 * state.Position());
       nextState.Velocity() = math::ClampRange(nextState.Velocity(),
         velocityMin, velocityMax);
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
   
       return std::pow(action.action[0], 2) * 0.1;
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
       state.Position() = math::Random(-0.6, -0.4);
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
     double positionMin;
   
     double positionMax;
   
     double positionGoal;
   
     double velocityMin;
   
     double velocityMax;
   
     double duration;
   
     double doneReward;
   
     size_t maxSteps;
   
     size_t stepsPerformed;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
