/**
 * @file methods/reinforcement_learning/async_learning.hpp
 * @author Shangtong Zhang
 *
 * This file is the definition of AsyncLearning class,
 * which is wrapper for various asynchronous learning algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_RL_ASYNC_LEARNING_HPP
#define MLPACK_METHODS_RL_ASYNC_LEARNING_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include "worker/worker.hpp"
#include "training_config.hpp"

namespace mlpack {

/**
 * Wrapper of various asynchronous learning algorithms,
 * e.g. async one-step Q-learning, async one-step Sarsa,
 * async n-step Q-learning and async advantage actor-critic.
 *
 * For more details, see the following:
 * @code
 * @inproceedings{mnih2016asynchronous,
 *   title     = {Asynchronous methods for deep reinforcement learning},
 *   author    = {Mnih, Volodymyr and Badia, Adria Puigdomenech and Mirza,
 *                Mehdi and Graves, Alex and Lillicrap, Timothy and Harley,
 *                Tim and Silver, David and Kavukcuoglu, Koray},
 *   booktitle = {International Conference on Machine Learning},
 *   pages     = {1928--1937},
 *   year      = {2016}
 * }
 * @endcode
 *
 * @tparam WorkerType The type of the worker.
 * @tparam EnvironmentType The type of reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
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
  /**
   * Construct an instance of the given async learning algorithm.
   *
   * @param config Hyper-parameters for training.
   * @param network The network model.
   * @param policy The behavior policy.
   * @param updater The optimizer.
   * @param environment The reinforcement learning task.
   */
  AsyncLearning(TrainingConfig config,
                NetworkType network,
                PolicyType policy,
                UpdaterType updater = UpdaterType(),
                EnvironmentType environment = EnvironmentType());

  /**
   * Starting async training.
   *
   * @tparam Measure The type of the measurement. It should be a
   *   callable object like
   *   @code
   *   bool foo(double reward);
   *   @endcode
   *   where reward is the total reward of a deterministic test episode,
   *   and the return value should indicate whether the training
   *   process is completed.
   * @param measure The measurement instance.
   */
  template <typename Measure>
  void Train(Measure& measure);

  //! Get training config.
  TrainingConfig& Config() { return config; }
  //! Modify training config.
  const TrainingConfig& Config() const { return config; }

  //! Get learning network.
  NetworkType& Network() { return learningNetwork; }
  //! Modify learning network.
  const NetworkType& Network() const { return learningNetwork; }

  //! Get behavior policy.
  PolicyType& Policy() { return policy; }
  //! Modify behavior policy.
  const PolicyType& Policy() const { return policy; }

  //! Get optimizer.
  UpdaterType& Updater() { return updater; }
  //! Modify optimizer.
  const UpdaterType& Updater() const { return updater; }

  //! Get the environment.
  EnvironmentType& Environment() { return environment; }
  //! Modify the environment.
  const EnvironmentType& Environment() const { return environment; }

 private:
  //! Locally-stored hyper-parameters.
  TrainingConfig config;

  //! Locally-stored global learning network.
  NetworkType learningNetwork;

  //! Locally-stored policy.
  PolicyType policy;

  //! Locally-stored optimizer.
  UpdaterType updater;

  //! Locally-stored task.
  EnvironmentType environment;
};

/**
 * Forward declaration of OneStepQLearningWorker.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
class OneStepQLearningWorker;

/**
 * Forward declaration of OneStepSarsaWorker.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
class OneStepSarsaWorker;

/**
 * Forward declaration of NStepQLearningWorker.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
class NStepQLearningWorker;

/**
 * Convenient typedef for async one step q-learning.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
using OneStepQLearning = AsyncLearning<OneStepQLearningWorker<EnvironmentType,
    NetworkType, UpdaterType, PolicyType>, EnvironmentType, NetworkType,
    UpdaterType, PolicyType>;

/**
 * Convenient typedef for async one step Sarsa.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
using OneStepSarsa = AsyncLearning<OneStepSarsaWorker<EnvironmentType,
    NetworkType, UpdaterType, PolicyType>, EnvironmentType, NetworkType,
    UpdaterType, PolicyType>;

/**
 * Convenient typedef for async n step q-learning.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
using NStepQLearning = AsyncLearning<NStepQLearningWorker<EnvironmentType,
    NetworkType, UpdaterType, PolicyType>, EnvironmentType, NetworkType,
    UpdaterType, PolicyType>;

} // namespace mlpack

// Include implementation
#include "async_learning_impl.hpp"

#endif
