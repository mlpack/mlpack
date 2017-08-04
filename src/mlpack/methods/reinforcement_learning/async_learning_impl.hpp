/**
 * @file async_learning_impl.hpp
 * @author Shangtong Zhang
 *
 * This file is the implementation of AsyncLearning class,
 * which is wrapper for various asynchronous learning algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
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
  /**
   * OpenMP doesn't support shared class member variables.
   * So we need to copy them to local variables.
   */
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

  // Synchronization for get/put task.
  #ifdef HAS_OPENMP
    omp_lock_t tasksLock;
    omp_init_lock(&tasksLock);
  #endif

  /**
   * Compute the number of threads for the for-loop. In general, we should use
   * OpenMP task rather than for-loop, here we do so to be compatible with some
   * compiler. We can switch to OpenMP task once MSVC supports OpenMP 3.0.
   */
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
    while (!stop)
    {
      // Assign task to current thread from queue.
      #ifdef HAS_OPENMP
        omp_set_lock(&tasksLock);
        if (tasks.empty())
        {
          omp_unset_lock(&tasksLock);
          continue;
        }
      #endif
      size_t task = tasks.front();
      tasks.pop();
      #ifdef HAS_OPENMP
        omp_unset_lock(&tasksLock);
      #endif

      // Get corresponding worker.
      WorkerType& worker = workers[task];
      double episodeReturn;
      if (worker.Step(learningNetwork, targetNetwork, totalSteps,
          policy, episodeReturn) && !task)
      {
        stop = measure(episodeReturn);
      }

      // Put task back to queue.
      #ifdef HAS_OPENMP
        omp_set_lock(&tasksLock);
      #endif
      tasks.push(task);
      #ifdef HAS_OPENMP
        omp_unset_lock(&tasksLock);
      #endif
    }
  }

  // Write back the learning network.
  this->learningNetwork = std::move(learningNetwork);
};

} // namespace rl
} // namespace mlpack

#endif

