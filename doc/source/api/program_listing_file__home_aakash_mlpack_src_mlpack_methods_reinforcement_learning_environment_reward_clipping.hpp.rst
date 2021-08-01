
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_reward_clipping.hpp:

Program Listing for File reward_clipping.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_environment_reward_clipping.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/environment/reward_clipping.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ENVIRONMENT_REWARD_CLIPPING_HPP
   #define MLPACK_METHODS_RL_ENVIRONMENT_REWARD_CLIPPING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/math/clamp.hpp>
   
   namespace mlpack {
   namespace rl {
   
   template <typename EnvironmentType>
   class RewardClipping
   {
    public:
     using State = typename EnvironmentType::State;
   
     using Action = typename EnvironmentType::Action;
   
     RewardClipping(EnvironmentType& environment,
                    const double minReward = -1.0,
                    const double maxReward = 1.0) :
       environment(environment),
       minReward(minReward),
       maxReward(maxReward)
     {
       // Nothing to do here
     }
   
     State InitialSample()
     {
       return environment.InitialSample();
     }
   
     bool IsTerminal(const State& state) const
     {
       return environment.IsTerminal(state);
     }
   
     double Sample(const State& state,
                   const Action& action,
                   State& nextState)
     {
       // Get original unclipped reward from base environment.
       double unclippedReward =  environment.Sample(state, action, nextState);
       // Clip rewards according to the min and max limit and return.
       return math::ClampRange(unclippedReward, minReward, maxReward);
     }
   
     double Sample(const State& state, const Action& action)
     {
       State nextState;
       return Sample(state, action, nextState);
     }
   
     EnvironmentType& Environment() const { return environment; }
     EnvironmentType& Environment() { return environment; }
   
     double MinReward() const { return minReward; }
     double& MinReward() { return minReward; }
   
     double MaxReward() const { return maxReward; }
     double& MaxReward() { return maxReward; }
   
    private:
     EnvironmentType environment;
   
     double minReward;
   
     double maxReward;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
