/**
 * @file async_learning.hpp
 * @author Shangtong Zhang
 *
 * This file is the definition of AsyncLearning class,
 * which implements asynchronous learning algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ASYNC_LEARNING_HPP
#define MLPACK_METHODS_RL_ASYNC_LEARNING_HPP

#include <mlpack/prereqs.hpp>
#include "worker/one_step_q_learning_worker.hpp"
#include "training_config.hpp"

namespace mlpack {
namespace rl {

/**
 * Implementation of various asynchronous learning algorithms,
 * such as async one-step Q-learning, async one-step Sarsa,
 * async n-step Q-learning and A3C
 */
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

  /**
   * Start learning from interaction with the environment
   */
  template <typename Measure>
  void Train(const Measure& measure);

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
using OneStepQLearning = AsyncLearning<OneStepQLearningWorker, EnvironmentType, NetworkType, UpdaterType, PolicyType>;

} // namespace rl
} // namespace mlpack

// Include implementation
#include "async_learning_impl.hpp"
#include "mlpack/methods/reinforcement_learning/worker/one_step_q_learning_worker.hpp"
#endif
