#ifndef MLPACK_METHODS_RL_DDPG_HPP
#define MLPACK_METHODS_RL_DDPG_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

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

  bool& Deterministic() { return &deterministic; }
  const bool& Deterministic() { return deterministic; }

 private:
  TrainingConfig& config;

  PolicyNetworkType& policyNetwork;
  PolicyNetworkType targetPolicyNetwork;
  
  QNetworkType& qNetwork;
  QNetworkType targetQNetwork;
  
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
  
  StateType state;
  ActionType action;

  size_t totalSteps;
  bool deterministic;

  mlpack::ann::MeanSquaredError<> lossFunction;
};

} // namespace rl
} // namespace mlpack

// Include implementation
#include "ddpg_impl.hpp"
#endif