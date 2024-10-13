#ifndef MLPACK_METHODS_RL_ALPHALIKE_IMPL_HPP
#define MLPACK_METHODS_RL_ALPHALIKE_IMPL_HPP

// #include "mlpack/prereqs.hpp"

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "alphazero.hpp"
#include "replay/replay.hpp"
#include "training_config.hpp"

namespace mlpack {

template <typename EnvironmentType, typename ValueNetworkType,
          typename PolicyNetworkType, typename UpdaterType, typename ReplayType>
AlphaZero<EnvironmentType, ValueNetworkType, PolicyNetworkType, UpdaterType,
          ReplayType>::AlphaZero(AlphaZeroConfig &config,
                                 ValueNetworkType &valueNetwork,
                                 PolicyNetworkType &policyNetwork,
                                 ReplayType &replayMethod,
                                 EnvironmentType environment,
                                 UpdaterType valueNetworkUpdater,
                                 UpdaterType policyNetworkUpdater)
    : config(config), valueNetwork(valueNetwork), policyNetwork(policyNetwork),
      replayMethod(replayMethod),
      valueNetworkUpdater(std::move(valueNetworkUpdater)),
#if ENS_VERSION_MAJOR >= 2
      valueNetworkUpdatePolicy(NULL),
#endif
      policyNetworkUpdater(std::move(policyNetworkUpdater)),
#if ENS_VERSION_MAJOR >= 2
      policyNetworkUpdatePolicy(NULL),
#endif
      environment(std::move(environment)), totalSteps(0),
      probaAction(ActionType::size), deterministic(false),
      r1(2.0 / (config.VMax() - config.VMin())),
      r2(-2.0 * config.VMin() / (config.VMax() - config.VMin()) - 1.0) {
  assert(config.SelfplaySteps() >= ActionType::size);
  // Reset all the networks.
  // Note: the value and policy networks have an if condition before reset.
  // This is because we don't want to reset a loaded(possibly pretrained) model
  // passed using this constructor.
  const size_t envSampleSize = environment.InitialSample().Encode().n_elem;

  if (policyNetwork.Parameters().n_elem != envSampleSize)
    policyNetwork.Reset(envSampleSize);

  if (valueNetwork.Parameters().n_elem != envSampleSize)
    valueNetwork.Reset(envSampleSize);

#if ENS_VERSION_MAJOR == 1
  this->valueNetworkUpdater.Initialize(valueNetwork.Parameters().n_rows,
                                       valueNetwork.Parameters().n_cols);
#else
  this->valueNetworkUpdatePolicy =
      new typename UpdaterType::template Policy<arma::mat, arma::mat>(
          this->valueNetworkUpdater, valueNetwork.Parameters().n_rows,
          valueNetwork.Parameters().n_cols);
#endif

#if ENS_VERSION_MAJOR == 1
  this->policyNetworkUpdater.Initialize(policyNetwork.Parameters().n_rows,
                                        policyNetwork.Parameters().n_cols);
#else
  this->policyNetworkUpdatePolicy =
      new typename UpdaterType::template Policy<arma::mat, arma::mat>(
          this->policyNetworkUpdater, policyNetwork.Parameters().n_rows,
          policyNetwork.Parameters().n_cols);
#endif
}

template <typename EnvironmentType, typename ValueNetworkType,
          typename PolicyNetworkType, typename UpdaterType, typename ReplayType>
AlphaZero<EnvironmentType, ValueNetworkType, PolicyNetworkType, UpdaterType,
          ReplayType>::~AlphaZero() {
  delete node;
  node = nullptr;
#if ENS_VERSION_MAJOR >= 2
  delete valueNetworkUpdatePolicy;
  delete policyNetworkUpdatePolicy;
#endif
}

template <typename EnvironmentType, typename ValueNetworkType,
          typename PolicyNetworkType, typename UpdaterType, typename ReplayType>
void AlphaZero<EnvironmentType, ValueNetworkType, PolicyNetworkType,
               UpdaterType, ReplayType>::SelfPlay() {
  node = new Node(state);
  for (size_t i = 0; i < config.SelfplaySteps(); ++i)
    Explore(node);
}

template <typename EnvironmentType, typename ValueNetworkType,
          typename PolicyNetworkType, typename UpdaterType, typename ReplayType>
void AlphaZero<EnvironmentType, ValueNetworkType, PolicyNetworkType,
               UpdaterType, ReplayType>::Update() {
  arma::mat sampledStates;
  arma::mat sampledProbaActions;
  arma::rowvec sampledReturns;
  arma::mat sampledNextStates;
  arma::irowvec isTerminal;

  replayMethod.Sample(sampledStates, sampledProbaActions, sampledReturns,
                      sampledNextStates, isTerminal);
  ScaleReturn(sampledReturns);

  arma::mat gradient;
  arma::mat gradientLoss;
  arma::mat stateValuesHat;

  valueNetwork.Forward(sampledNextStates, stateValuesHat);
  lossValueNetwork.Backward(stateValuesHat, sampledReturns, gradientLoss);
  valueNetwork.Backward(sampledNextStates, gradientLoss, gradient);

#if ENS_VERSION_MAJOR == 1
  valueNetworkUpdater.Update(valueNetwork.Parameters(), config.StepSize(),
                             gradient);
#else
  valueNetworkUpdatePolicy->Update(valueNetwork.Parameters(), config.StepSize(),
                                   gradient);
#endif

  arma::mat actionLogProbaHat;
  policyNetwork.Forward(sampledStates, actionLogProbaHat);
  lossPolicyNetwork.Backward(actionLogProbaHat, sampledProbaActions,
                             gradientLoss);
  policyNetwork.Backward(sampledStates, gradientLoss, gradient);

#if ENS_VERSION_MAJOR == 1
  policyNetworkUpdater.Update(policyNetwork.Parameters(), config.StepSize(),
                              gradient);
#else
  policyNetworkUpdatePolicy->Update(policyNetwork.Parameters(),
                                    config.StepSize(), gradient);
#endif

  double loss_v = lossValueNetwork.Forward(stateValuesHat, sampledReturns);
  double loss_p =
      lossPolicyNetwork.Forward(actionLogProbaHat, sampledProbaActions);
  std::cout << "losses: v=" << loss_v << ", p=" << loss_p << "\n";
}

template <typename EnvironmentType, typename ValueNetworkType,
          typename PolicyNetworkType, typename UpdaterType, typename ReplayType>
void AlphaZero<EnvironmentType, ValueNetworkType, PolicyNetworkType,
               UpdaterType, ReplayType>::ScaleReturn(arma::mat
                                                         &sampledReturns) {
  sampledReturns *= r1;
  sampledReturns += r2;
}

template <typename EnvironmentType, typename ValueNetworkType,
          typename PolicyNetworkType, typename UpdaterType, typename ReplayType>
double AlphaZero<EnvironmentType, ValueNetworkType, PolicyNetworkType,
                 UpdaterType, ReplayType>::Episode() {
  // Get the initial state from environment.
  state = environment.InitialSample();

  // Track the steps in this episode.
  size_t steps = 0;

  // Track the return of this episode.
  double totalReturn = 0.0;

  // Running until get to the terminal state.
  while (!environment.IsTerminal(state)) {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;
    /* Play simulated games and update nodes’ ucb scores */
    SelfPlay();
    /** True play:
     *  Next modifies the node, the action action and probaAction
     */
    Next(node, action, probaAction);
    /* Interact with the environment to advance to next state */
    StateType nextState;
    double reward = environment.Sample(state, action, nextState);
    totalReturn += reward;
    steps++;
    totalSteps++;
    /* Store the transition for replay */
    replayMethod.Store(state, probaAction, reward, nextState,
                       environment.IsTerminal(nextState), config.Discount());
    /* Update current state */
    state = nextState;
  }
  /* Network update */
  if ((!deterministic) && (totalSteps >= config.ExplorationSteps())) {
    for (size_t i = 0; i < config.UpdateInterval(); i++)
      Update();
  }
  /* Delete the node created at the start of the episode */
  delete node;
  node = nullptr;

  return totalReturn;
}

} // namespace mlpack

#endif
