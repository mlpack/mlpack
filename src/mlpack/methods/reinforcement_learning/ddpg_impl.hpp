#ifndef MLPACK_METHODS_RL_DDPG_IMPL_HPP
#define MLPACK_METHODS_RL_DDPG_IMPL_HPP

#include <mlpack/prereqs.hpp>

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
  // Get the deterministic action at current state from policy
  arma::colvec outputAction;
  policyNetwork.Predict(state.Encode(), outputAction);

  // Add noise to convert the deterministic policy into exploration policy
  arma::colvec noise = arma::randn<arma::colvec>(outputAction.n_rows) * 0.1;
  noise = arma::clamp(noise, -0.25, 0.25);
  outputAction = outputAction + noise;

  action.action = arma::conv_to<decltype(action.action)>::from(outputAction);
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
  // Sample from previous experience.
  arma::mat sampledStates;
  std::vector<ActionType> sampledActionsVector;
  arma::rowvec sampledRewards;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
      sampledNextStates, isTerminal);

  /*
    Critic network update.
  */

  // Predict next actions using target policy given sampled next states.
  arma::mat predictedNextActions;
  targetPolicyNetwork.Predict(sampledNextStates, predictedNextActions);

  // Compute Q target values given sampled next states and predicted next actions.
  arma::rowvec qTarget;
  arma::mat qTargetInput = arma::join_vert(predictedNextActions, sampledNextStates);
  targetQNetwork.Predict(qTargetInput, qTarget);

  // Compute TD target.
  arma::rowvec tdTarget = sampledRewards + config.Discount() * (
      (1 - isTerminal) * qTarget);

  // Compute Q values given sampled current states and sampled current actions.
  arma::mat sampledActions(action.size, sampledActionsVector.size());
  for (auto& a: sampledActionsVector)
    sampledActions.col(i) = arma::conv_to<arma::colvec>::from(a.action);
  
  arma::mat Q;
  arma::mat qInput = arma::join_vert(sampledActions, sampledStates);
  qNetwork.Forward(qInput, Q);

  // Compute gradient.
  arma::mat gradLoss, gradQ;
  lossFunction.Backward(Q, tdTarget, gradLoss);
  qNetwork.Backward(qInput, gradLoss, gradQ);

  // Update by applying gradient.
  qNetworkUpdater.Update(qNetwork.Parameters(), config.StepSize(), gradQ);

  /*
    Actor network update.
  */

  // Compute gradient for each samples and then take the average.
  arma::mat gradient;
  for (size_t i = 0; i < sampledStates.n_cols; i++)
  {
    arma::colvec state = sampledStates.col(i);

    // Predict action using true policy given sampled current state.
    arma::colvec predictedAction;
    policyNetwork.Forward(state, predictedAction);

    // Compute Q value given sampled current state and predicted action.
    qInput = arma::join_vert(predictedAction, state);
    qNetwork.Forward(qInput, Q);

    // Compute gradient.
    qNetwork.Backward()
    policyNetwork.Backward(state, )

    if (i == 0)
    {
      gradient.copy_size(grad);
      gradient.fill(0.0);
    }

    gradient += grad;
  }
  gradient /= sampledStates.n_cols;
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