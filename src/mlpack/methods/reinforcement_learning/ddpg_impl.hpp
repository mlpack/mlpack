#ifndef MLPACK_METHODS_RL_DDPG_IMPL_HPP
#define MLPACK_METHODS_RL_DDPG_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "ddpg.hpp"

namespace mlpack {
namespace rl {

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
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
    #if ENS_VERSION_MAJOR >= 2
    qNetworkUpdatePolicy(NULL),
    #endif
    policyNetworkUpdater(std::move(policyNetworkUpdater)),
    #if ENS_VERSION_MAJOR >= 2
    policyNetworkUpdatePolicy(NULL),
    #endif
    environment(std::move(environment)),
    totalSteps(0),
    deterministic(false)
{
  if (qNetwork.Parameters().is_empty())
    qNetwork.ResetParameters();
  if (policyNetwork.Parameters().is_empty())
    policyNetwork.ResetParameters();
  
  targetQNetwork = qNetwork;
  targetPolicyNetwork = policyNetwork;

  #if ENS_VERSION_MAJOR == 1
  qNetworkUpdater.Initialize(qNetwork.Parameters().n_rows,
                             qNetwork.Parameters().n_cols);
  #else
  qNetworkUpdatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->qNetworkUpdater,
                                   qNetwork.Parameters().n_rows,
                                   qNetwork.Parameters().n_cols);
  #endif

  #if ENS_VERSION_MAJOR == 1
  policyNetworkUpdater.Initialize(policyNetwork.Parameters().n_rows,
                                  policyNetwork.Parameters().n_cols);
  #else
  policyNetworkUpdatePolicy = new typename UpdaterType::template
      Policy<arma::mat, arma::mat>(this->policyNetworkUpdater,
                                   policyNetwork.Parameters().n_rows,
                                   policyNetwork.Parameters().n_cols);
  #endif
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::~DDPG()
{
  #if ENS_VERSION_MAJOR >= 2
  delete qNetworkUpdatePolicy;
  delete policyNetworkUpdatePolicy;
  #endif
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::SelectAction()
{
  // Get the deterministic action at current state from policy
  arma::colvec outputAction;
  policyNetwork.Predict(state.Encode(), outputAction);

  // Add noise to convert the deterministic policy into exploration policy
  if (!deterministic)
  {
    arma::colvec noise = arma::randn<arma::colvec>(outputAction.n_rows) * 0.1;
    noise = arma::clamp(noise, -0.25, 0.25);
    outputAction = outputAction + noise;
  }

  action.action = arma::conv_to<decltype(action.action)>::from(outputAction);
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
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
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
void DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::TrainAgent()
{
  // Sample from previous experience.
  arma::mat sampledStates;
  std::vector<ActionType> sampledActionsVector;
  arma::rowvec sampledRewards;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActionsVector,
      sampledRewards, sampledNextStates, isTerminal);

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
      (1 - isTerminal) % qTarget);

  // Compute Q values given sampled current states and sampled current actions.
  arma::mat sampledActions(action.size, sampledActionsVector.size());
  for (size_t i = 0; i < sampledActionsVector.size(); i++)
    sampledActions.col(i) = arma::conv_to<arma::colvec>::from(
      sampledActionsVector[i].action);
  
  arma::mat Q;
  arma::mat qInput = arma::join_vert(sampledActions, sampledStates);
  qNetwork.Forward(qInput, Q);

  // Compute gradient.
  arma::mat gradLoss, gradQ;
  lossFunction.Backward(Q, tdTarget, gradLoss);
  qNetwork.Backward(qInput, gradLoss, gradQ);

  // Update by applying gradient.
  #if ENS_VERSION_MAJOR == 1
  qNetworkUpdater.Update(qNetwork.Parameters(), config.StepSize(), gradQ);
  #else
  qNetworkUpdatePolicy->Update(qNetwork.Parameters(), config.StepSize(), gradQ);
  #endif

  /*
    Actor network update.
  */

  // Get first hidden layer weight corresponding to action input
  // to compute grad of qNetwork w.r.t action.
  size_t actionSize = predictedNextActions.n_rows;
  size_t firstHiddenSize = boost::get<mlpack::ann::Linear<> *>
      (qNetwork.Model()[0])->OutputSize();

  arma::mat firstHiddenWeightAction = arma::reshape(
      qNetwork.Parameters().rows(0, firstHiddenSize * actionSize - 1),
      firstHiddenSize, actionSize);

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

    // Compute gradient of qNetwork w.r.t action.
    // Use -1 to change grad ascent into grad descent.
    qNetwork.Backward(qInput, -1, gradQ);
    
    arma::colvec gradQBias = gradQ(qInput.n_rows * firstHiddenSize, 0,
        arma::size(firstHiddenSize, 1));
    arma::colvec gradQwrtAction = firstHiddenWeightAction.t() * gradQBias;

    // Compute the gradient of policyNetwork.
    arma::mat gradPolicy;
    policyNetwork.Backward(state, gradQwrtAction, gradPolicy);

    if (i == 0)
    {
      gradient.copy_size(gradPolicy);
      gradient.fill(0.0);
    }

    gradient += gradPolicy;
  }
  gradient /= sampledStates.n_cols;

  // Update by applying gradient.
  #if ENS_VERSION_MAJOR == 1
  policyNetworkUpdater.Update(policyNetwork.Parameters(),
      config.StepSize(), gradient);
  #else
  policyNetworkUpdatePolicy->Update(policyNetwork.Parameters(),
      config.StepSize(), gradient);
  #endif
  
  // Soft-update target network.
  SoftUpdateTargetNetwork(config.Rho());
}

template <
  typename EnvironmentType,
  typename QNetworkType,
  typename PolicyNetworkType,
  typename UpdaterType,
  typename ReplayType
>
double DDPG<
  EnvironmentType,
  QNetworkType,
  PolicyNetworkType,
  UpdaterType,
  ReplayType
>::Episode()
{
  state = environment.InitialSample();
  size_t steps= 0;
  double totalReturn = 0.;

  while (!environment.IsTerminal(state)) {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;
    
    SelectAction();

    StateType nextState;
    double reward = environment.Sample(state, action, nextState);

    totalReturn += reward;
    steps++;
    totalSteps++;

    replayMethod.Store(state, action, reward, nextState,
      environment.IsTerminal(nextState), config.Discount());

    state = nextState;

    // Make sure to populate the replay buffer before sampling.
    if (deterministic || totalSteps < config.ExplorationSteps())
      continue;
    
    TrainAgent();
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack

#endif