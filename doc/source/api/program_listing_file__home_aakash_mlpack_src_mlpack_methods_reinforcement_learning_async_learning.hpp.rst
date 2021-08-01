
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_async_learning.hpp:

Program Listing for File async_learning.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_async_learning.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/async_learning.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ASYNC_LEARNING_HPP
   #define MLPACK_METHODS_RL_ASYNC_LEARNING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "worker/one_step_q_learning_worker.hpp"
   #include "worker/one_step_sarsa_worker.hpp"
   #include "worker/n_step_q_learning_worker.hpp"
   #include "training_config.hpp"
   
   namespace mlpack {
   namespace rl {
   
   template <
     typename WorkerType,
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   class AsyncLearning
   {
    public:
     AsyncLearning(TrainingConfig config,
                   NetworkType network,
                   PolicyType policy,
                   UpdaterType updater = UpdaterType(),
                   EnvironmentType environment = EnvironmentType());
   
     template <typename Measure>
     void Train(Measure& measure);
   
     TrainingConfig& Config() { return config; }
     const TrainingConfig& Config() const { return config; }
   
     NetworkType& Network() { return learningNetwork; }
     const NetworkType& Network() const { return learningNetwork; }
   
     PolicyType& Policy() { return policy; }
     const PolicyType& Policy() const { return policy; }
   
     UpdaterType& Updater() { return updater; }
     const UpdaterType& Updater() const { return updater; }
   
     EnvironmentType& Environment() { return environment; }
     const EnvironmentType& Environment() const { return environment; }
   
    private:
     TrainingConfig config;
   
     NetworkType learningNetwork;
   
     PolicyType policy;
   
     UpdaterType updater;
   
     EnvironmentType environment;
   };
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   class OneStepQLearningWorker;
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   class OneStepSarsaWorker;
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   class NStepQLearningWorker;
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   using OneStepQLearning = AsyncLearning<OneStepQLearningWorker<EnvironmentType,
       NetworkType, UpdaterType, PolicyType>, EnvironmentType, NetworkType,
       UpdaterType, PolicyType>;
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   using OneStepSarsa = AsyncLearning<OneStepSarsaWorker<EnvironmentType,
       NetworkType, UpdaterType, PolicyType>, EnvironmentType, NetworkType,
       UpdaterType, PolicyType>;
   
   template <
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   using NStepQLearning = AsyncLearning<NStepQLearningWorker<EnvironmentType,
       NetworkType, UpdaterType, PolicyType>, EnvironmentType, NetworkType,
       UpdaterType, PolicyType>;
   
   } // namespace rl
   } // namespace mlpack
   
   // Include implementation
   #include "async_learning_impl.hpp"
   
   #endif
