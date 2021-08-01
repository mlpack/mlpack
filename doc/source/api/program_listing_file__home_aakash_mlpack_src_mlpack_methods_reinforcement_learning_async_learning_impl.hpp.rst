
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_async_learning_impl.hpp:

Program Listing for File async_learning_impl.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_async_learning_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/async_learning_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_ASYNC_LEARNING_IMPL_HPP
   #define MLPACK_METHODS_RL_ASYNC_LEARNING_IMPL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "queue"
   
   namespace mlpack {
   namespace rl {
   
   template <
     typename WorkerType,
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   AsyncLearning<
     WorkerType,
     EnvironmentType,
     NetworkType,
     UpdaterType,
     PolicyType
   >::AsyncLearning(
       TrainingConfig config,
       NetworkType network,
       PolicyType policy,
       UpdaterType updater,
       EnvironmentType environment):
       config(std::move(config)),
       learningNetwork(std::move(network)),
       policy(std::move(policy)),
       updater(std::move(updater)),
       environment(std::move(environment))
   { /* Nothing to do here. */ };
   
   template <
     typename WorkerType,
     typename EnvironmentType,
     typename NetworkType,
     typename UpdaterType,
     typename PolicyType
   >
   template <typename Measure>
   void AsyncLearning<
     WorkerType,
     EnvironmentType,
     NetworkType,
     UpdaterType,
     PolicyType
   >::Train(Measure& measure)
   {
     NetworkType learningNetwork = std::move(this->learningNetwork);
     if (learningNetwork.Parameters().is_empty())
       learningNetwork.ResetParameters();
     NetworkType targetNetwork = learningNetwork;
     size_t totalSteps = 0;
     PolicyType policy = this->policy;
     bool stop = false;
   
     // Set up worker pool, worker 0 will be deterministic for evaluation.
     std::vector<WorkerType> workers;
     for (size_t i = 0; i <= config.NumWorkers(); ++i)
     {
       workers.push_back(WorkerType(updater, environment, config, !i));
       workers.back().Initialize(learningNetwork);
     }
     // Set up task queue corresponding to worker pool.
     std::queue<size_t> tasks;
     for (size_t i = 0; i <= config.NumWorkers(); ++i)
       tasks.push(i);
   
     size_t numThreads = 0;
     #pragma omp parallel reduction(+:numThreads)
     numThreads++;
     Log::Debug << numThreads << " threads will be used in total." << std::endl;
   
     #pragma omp parallel for shared(stop, workers, tasks, learningNetwork, \
         targetNetwork, totalSteps, policy)
     for (omp_size_t i = 0; i < numThreads; ++i)
     {
       #pragma omp critical
       {
         #ifdef HAS_OPENMP
           Log::Debug << "Thread " << omp_get_thread_num() <<
               " started." << std::endl;
         #endif
       }
       size_t task = std::numeric_limits<size_t>::max();
       while (!stop)
       {
         // Assign task to current thread from queue.
         #pragma omp critical
         {
           if (task != std::numeric_limits<size_t>::max())
             tasks.push(task);
   
           if (!tasks.empty())
           {
             task = tasks.front();
             tasks.pop();
           }
         };
   
         // This may happen when threads are more than workers.
         if (task == std::numeric_limits<size_t>::max())
           continue;
   
         // Get corresponding worker.
         WorkerType& worker = workers[task];
         double episodeReturn;
         if (worker.Step(learningNetwork, targetNetwork, totalSteps,
             policy, episodeReturn) && !task)
         {
           stop = measure(episodeReturn);
         }
       }
     }
   
     // Write back the learning network.
     this->learningNetwork = std::move(learningNetwork);
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
   
