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
>::AsyncLearning(TrainingConfig config,
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
  bool stop = false;
  size_t totalSteps = 0;
  PolicyType policy = this->policy;

  // Make sure OpenMP will start desired number of threads.
  omp_set_dynamic(0);
  // An extra thread for test.
  omp_set_num_threads(config.NumOfWorkers() + 1);

  // All the variables that doesn't show up in this list will be copied.
  #pragma omp parallel for shared(learningNetwork, targetNetwork, \
      stop, totalSteps, policy)
  for (size_t i = 0; i <= config.NumOfWorkers(); ++i)
  {
    while (!stop)
    {
      if (i < config.NumOfWorkers())
      {
        WorkerType::Episode(learningNetwork, targetNetwork, stop, totalSteps,
            policy, updater, environment, config, false);
      }
      else
      {
        stop = measure(WorkerType::Episode(learningNetwork, targetNetwork, stop,
            totalSteps, policy, updater, environment, config, true));
      }
    }
  }

  // Write back the learning network.
  this->learningNetwork = std::move(learningNetwork);
};

} // namespace rl
} // namespace mlpack

#endif

