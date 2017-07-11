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
        AsyncLearning(NetworkType network,
                      const double stepSize,
                      const double discount,
                      const size_t nWorkers,
                      const size_t updateInterval,
                      const size_t targetNetworkSyncInterval,
                      const size_t stepLimit,
                      PolicyType policy,
                      UpdaterType updater,
                      const double gradientLimit,
                      EnvironmentType environment):
        learningNetwork(std::move(network)),
        stepSize(stepSize), discount(discount),
        nWorkers(nWorkers),
        updateInterval(updateInterval),
        targetNetworkSyncInterval(targetNetworkSyncInterval),
        stepLimit(stepLimit),
        policy(std::move(policy)),
        updater(std::move(updater)),
        gradientLimit(gradientLimit),
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
  omp_set_num_threads(nWorkers + 1);
  #pragma omp parallel for shared(learningNetwork, targetNetwork, stop, totalSteps, policy)
  for (size_t i = 0; i <= nWorkers; ++i)
  {
    while (!stop)
    {
      if (i < nWorkers)
      {
        double reward = Episode(learningNetwork, targetNetwork, stop, totalSteps, policy, false, WorkerType());
        if (i == 0)
          Log::Debug << "Worker 0 episode reward: " << reward << std::endl;
      }
      else
      {
        double reward = Episode(learningNetwork, targetNetwork, stop, totalSteps, policy, true, WorkerType());
        stop = measure(reward);
      }
    }
  }
  this->learningNetwork = std::move(learningNetwork);
};

}
}

#endif

