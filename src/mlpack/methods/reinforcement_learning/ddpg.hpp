#ifndef MLPACK_METHODS_RL_DDPG_HPP
#define MLPACK_METHODS_RL_DDPG_HPP

#include <vector>
#include "ra_util.hpp"
#include <training_config.hpp>
#include "training_config.hpp"
#include "replay/random_replay.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType = RandomReplay<EnvironmentType>
>
class DDPG
{
 public:
  using StateType = typename EnvironmentType::State;
  using ActionType = typename EnvironmentType::Action;

  DDPG(TrainingConfig& config,
       QNetworkType& qNetwork,
       PolicyNetworkType& policyNetwork,
       ReplayType& replayMethod,
       UpdaterType qNetworkUpdater = UpdaterType(),
       UpdaterType policyNetworkUpdater = UpdaterType(),
       EnvironmentType environment = EnvironmentType());
  
  void SelectAction();

  void SoftUpdateTargetNetwork(double tau);

  void TrainAgent();
  
  double Episode();

 private:
  TrainingConfig& config;

  PolicyNetworkType& policyNetwork;
  PolicyNetworkType targetPolicyNetwork;
  
  QNetworkType& qNetwork;
  QNetworkType targetQNetwork;
  
  ReplayType& replayMethod;

  UpdaterType qNetworkUpdater;
  UpdaterType policyNetworkUpdater;

  EnvironmentType environment;
  
  StateType state;
  ActionType action;

  size_t totalSteps;

  mlpack::ann::MeanSquaredError<> lossFunction;
};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "ddpg_impl.hpp"
#endif