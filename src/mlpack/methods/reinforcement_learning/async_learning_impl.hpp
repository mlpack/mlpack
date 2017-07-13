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
AsyncLearning<WorkerType, EnvironmentType, NetworkType, UpdaterType, PolicyType>::
        AsyncLearning(TrainingConfig config,
                      NetworkType network,
                      PolicyType policy,
                      UpdaterType updater,
                      EnvironmentType environment):
        config(std::move(config)),
        learningNetwork(std::move(network)),
        policy(std::move(policy)),
        updater(std::move(updater)),
        environment(std::move(environment))
        {};

template <
        typename WorkerType,
        typename EnvironmentType,
        typename NetworkType,
        typename UpdaterType,
        typename PolicyType
>
template <typename Measure>
void AsyncLearning<WorkerType, EnvironmentType, NetworkType, UpdaterType, PolicyType>::
       Train(const Measure& measure)
{
  NetworkType learningNetwork = std::move(this->learningNetwork);
  learningNetwork.ResetParameters();
  NetworkType targetNetwork = learningNetwork;
  bool stop = false;
  size_t totalSteps = 0;
  PolicyType policy = this->policy;

  omp_set_dynamic(0);
  omp_set_num_threads(config.NumOfWorkers() + 1);
  #pragma omp parallel for shared(learningNetwork, targetNetwork, stop, totalSteps, policy)
  for (size_t i = 0; i <= config.NumOfWorkers(); ++i)
  {
    while (!stop)
    {
      if (i < config.NumOfWorkers())
        WorkerType::Episode(learningNetwork, targetNetwork, stop, totalSteps, policy, updater, environment, config, false);
      else
        stop = measure(WorkerType::Episode(learningNetwork, targetNetwork, stop, totalSteps, policy, updater, environment, config, true));
    }
  }
  this->learningNetwork = std::move(learningNetwork);
};

}
}

#endif

