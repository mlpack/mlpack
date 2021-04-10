#ifndef MLPACK_METHODS_RL_DDPG_IMPL_HPP
#define MLPACK_METHODS_RL_DDPG_IMPL_HPP

#include "ddpg.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
DDPG<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::DDPG(TrainingConfig& config,
        QNetworkType& qNetwork,
        PolicyNetworkType& policyNetwork,
        ReplayType& replayMethod,
        UpdaterType qNetworkUpdater,
        UpdaterType policyNetworkUpdater,
        EnvironmentType environment):
    config(config),
    qNetwork(qNetwork),
    policyNetwork(policyNetwork),
    replayMethod(replayMethod),
    qNetworkUpdater(std::move(qNetworkUpdater)),
    policyNetworkUpdater(std::move(policyNetworkUpdater)),
    environment(std::move(environment))
{
  if (qNetwork.Parameters().is_empty())
    qNetwork.ResetParameters();
  if (policyNetwork.Parameters().is_empty())
    policyNetwork.ResetParameters();
  
  targetQNetwork = qNetwork;
  targetPolicyNetwork = policyNetwork;

  qNetworkUpdater.Initialize(qNetwork.Parameters().n_rows,
                             qNetwork.Parameters().n_cols);
  policyNetworkUpdater.Initialize(policyNetwork.Parameters().n_rows,
                                  policyNetwork.Parameters().n_cols);
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::SelectAction()
{
  arma::colvec outputAction;
  policyNetwork.Predict(state.Encode(), outputAction);
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::SoftUpdateTargetNetwork(double tau)
{
  targetQNetwork.Parameters() = tau * qNetwork.Parameters() + 
      (1 - tau) * targetQNetwork.Parameters();
  targetPolicyNetwork.Parameters() = tau * policyNetwork.Parameters() +
      (1 - tau) * targetPolicyNetwork.Parameters();
}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::TrainAgent()
{

}

template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double DDPG<
  EnvironmentType,
  NetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Episode()
{
  state = environment.InitialSample();
  size_t steps= 0;
  double totalReturn = 0.;

  while (!environment.IsTerminal(state)) {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;
    
    this->SelectAction();

    StateType nextState;
    double reward = environment.Sample(state, action, nextState);

    totalReturn += reward;
    steps++;
    totalSteps++;

    replayMethod.Store(state, action, reward, nextState,
      environment.IsTerminal(nextState), config.Discount());

    state = nextState;

    // Make sure to populate the replay buffer before sampling.
    if (totalSteps < config.ExplorationSteps())
      continue;
    
    this->TrainAgent();
  }
}

} // namespace rl
} // namespace mlpack

#endif