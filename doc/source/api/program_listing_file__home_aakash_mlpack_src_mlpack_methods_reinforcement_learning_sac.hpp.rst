
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_sac.hpp:

Program Listing for File sac.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_sac.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/sac.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_SAC_HPP
   #define MLPACK_METHODS_RL_SAC_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "replay/random_replay.hpp"
   #include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
   #include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
   #include <mlpack/methods/ann/visitor/parameters_visitor.hpp>
   #include "training_config.hpp"
   
   namespace mlpack {
   namespace rl {
   
   template <
     typename EnvironmentType,
     typename QNetworkType,
     typename PolicyNetworkType,
     typename UpdaterType,
     typename ReplayType = RandomReplay<EnvironmentType>
   >
   class SAC
   {
    public:
     using StateType = typename EnvironmentType::State;
   
     using ActionType = typename EnvironmentType::Action;
   
     SAC(TrainingConfig& config,
         QNetworkType& learningQ1Network,
         PolicyNetworkType& policyNetwork,
         ReplayType& replayMethod,
         UpdaterType qNetworkUpdater = UpdaterType(),
         UpdaterType policyNetworkUpdater = UpdaterType(),
         EnvironmentType environment = EnvironmentType());
   
     ~SAC();
   
     void SoftUpdate(double rho);
   
     void Update();
   
     void SelectAction();
   
     double Episode();
   
     size_t& TotalSteps() { return totalSteps; }
     const size_t& TotalSteps() const { return totalSteps; }
   
     StateType& State() { return state; }
     const StateType& State() const { return state; }
   
     const ActionType& Action() const { return action; }
   
     bool& Deterministic() { return deterministic; }
     const bool& Deterministic() const { return deterministic; }
   
   
    private:
     TrainingConfig& config;
   
     QNetworkType& learningQ1Network;
     QNetworkType learningQ2Network;
   
     QNetworkType targetQ1Network;
     QNetworkType targetQ2Network;
   
     PolicyNetworkType& policyNetwork;
   
     ReplayType& replayMethod;
   
     UpdaterType qNetworkUpdater;
     #if ENS_VERSION_MAJOR >= 2
     typename UpdaterType::template Policy<arma::mat, arma::mat>*
         qNetworkUpdatePolicy;
     #endif
   
     UpdaterType policyNetworkUpdater;
     #if ENS_VERSION_MAJOR >= 2
     typename UpdaterType::template Policy<arma::mat, arma::mat>*
         policyNetworkUpdatePolicy;
     #endif
   
     EnvironmentType environment;
   
     size_t totalSteps;
   
     StateType state;
   
     ActionType action;
   
     bool deterministic;
   
     mlpack::ann::MeanSquaredError<> lossFunction;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   // Include implementation
   #include "sac_impl.hpp"
   #endif
