
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_learning.hpp:

Program Listing for File q_learning.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_learning.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/q_learning.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_Q_LEARNING_HPP
   #define MLPACK_METHODS_RL_Q_LEARNING_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "replay/random_replay.hpp"
   #include "replay/prioritized_replay.hpp"
   #include "training_config.hpp"
   
   namespace mlpack {
   namespace rl {
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType,
     typename ReplayType = RandomReplay<EnvironmentType>
   >
   class QLearning
   {
    public:
     using StateType = typename EnvironmentType::State;
   
     using ActionType = typename EnvironmentType::Action;
   
     QLearning(TrainingConfig& config,
               NetworkType& network,
               PolicyType& policy,
               ReplayType& replayMethod,
               UpdaterType updater = UpdaterType(),
               EnvironmentType environment = EnvironmentType());
   
     ~QLearning();
   
     void TrainAgent();
   
     void TrainCategoricalAgent();
   
     void SelectAction();
   
     double Episode();
   
     size_t& TotalSteps() { return totalSteps; }
     const size_t& TotalSteps() const { return totalSteps; }
   
     StateType& State() { return state; }
     const StateType& State() const { return state; }
   
     const ActionType& Action() const { return action; }
   
     EnvironmentType& Environment() { return environment; }
     const EnvironmentType& Environment() const { return environment; }
   
     bool& Deterministic() { return deterministic; }
     const bool& Deterministic() const { return deterministic; }
   
     const NetworkType& Network() const { return learningNetwork; }
     NetworkType& Network() { return learningNetwork; }
   
    private:
     arma::Col<size_t> BestAction(const arma::mat& actionValues);
   
     TrainingConfig& config;
   
     NetworkType& learningNetwork;
   
     NetworkType targetNetwork;
   
     PolicyType& policy;
   
     ReplayType& replayMethod;
   
     UpdaterType updater;
     #if ENS_VERSION_MAJOR >= 2
     typename UpdaterType::template Policy<arma::mat, arma::mat>* updatePolicy;
     #endif
   
     EnvironmentType environment;
   
     size_t totalSteps;
   
     StateType state;
   
     ActionType action;
   
     bool deterministic;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   // Include implementation
   #include "q_learning_impl.hpp"
   #endif
