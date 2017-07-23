/**
 * @file one_step_q_learning_worker.hpp
 * @author Shangtong Zhang
 *
 * This file is the definition of OneStepQLearningWorker class,
 * which implements an episode for async one step Q-Learning algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_WORKER_ONE_STEP_Q_LEARNING_WORKER_HPP
#define MLPACK_METHODS_RL_WORKER_ONE_STEP_Q_LEARNING_WORKER_HPP

#include <mlpack/methods/reinforcement_learning/training_config.hpp>

namespace mlpack {
namespace rl {

class OneStepQLearningWorker
{
 public:
  /**
   * An episode of one step Q-Learning.
   *
   * @tparam EnvironmentType The type of the reinforcement learning task.
   * @tparam NetworkType The type of the network model.
   * @tparam UpdaterType The type of the optimizer.
   * @tparam PolicyType The type of the behavior policy.
   * @param learningNetwork Global learning network.
   * @param targetNetwork Global target network.
   * @param stop Whether training is end of not.
   * @param totalSteps Total time steps of all the workers
   * @param policy Behavior policy.
   * @param updater Local optimizer of the worker.
   * @param environment The reinforcement learning task.
   * @param config Hyper-parameters for training.
   * @param deterministic Whether this episode is deterministic or not.
   * @return Total reward of this episode.
   */
  template <
    typename EnvironmentType,
    typename NetworkType,
    typename UpdaterType,
    typename PolicyType
  >
  static double Episode(NetworkType& learningNetwork,
                        NetworkType& targetNetwork,
                        bool& stop,
                        size_t& totalSteps,
                        PolicyType& policy,
                        UpdaterType& updater,
                        EnvironmentType& environment,
                        const TrainingConfig& config,
                        bool deterministic) {
    using StateType = typename EnvironmentType::State;
    using ActionType = typename EnvironmentType::Action;
    using TransitionType = std::tuple<StateType, ActionType, double, StateType>;

    updater.Initialize(learningNetwork.Parameters().n_rows,
        learningNetwork.Parameters().n_cols);

    // Construct the local network from global network.
    NetworkType network = learningNetwork;

    // Get initial state.
    StateType state = environment.InitialSample();

    size_t steps = 0;
    double totalReturn = 0.0;

    // Store pending transitions for update later on.
    std::vector<TransitionType> pending(config.UpdateInterval());
    size_t pendingIndex = 0;

    while (!stop && (!config.StepLimit() || steps < config.StepLimit())) {
      // Interact with the environment.
      arma::colvec actionValue;
      network.Predict(state.Encode(), actionValue);
      ActionType action = policy.Sample(actionValue, deterministic);
      StateType nextState;
      double reward = environment.Sample(state, action, nextState);
      bool terminal = environment.IsTerminal(nextState);

      totalReturn += reward;
      steps++;

      if (deterministic)
      {
        if (terminal)
          break;
        state = nextState;
        continue;
      }

      #pragma omp atomic
      totalSteps++;

      pending[pendingIndex] = std::make_tuple(state, action, reward, nextState);
      pendingIndex++;

      // Do update.
      if (terminal || pendingIndex >= config.UpdateInterval())
      {
        // Initialize the gradient storage.
        arma::mat totalGradients(learningNetwork.Parameters().n_rows,
            learningNetwork.Parameters().n_cols);
        for (size_t i = 0; i < pending.size(); ++i) {
          TransitionType &transition = pending[i];

          // Compute the target state-action value.
          arma::colvec actionValue;
          #pragma omp critical
          {
            targetNetwork.Predict(
                std::get<3>(transition).Encode(), actionValue);
          };
          double targetActionValue = actionValue.max();
          if (terminal && i == pending.size() - 1)
            targetActionValue = 0;
          targetActionValue = std::get<2>(transition) +
              config.Discount() * targetActionValue;

          // Compute the training target for current state.
          network.Forward(std::get<0>(transition).Encode(), actionValue);
          actionValue[std::get<1>(transition)] = targetActionValue;

          // Compute gradient.
          arma::mat gradients;
          network.Backward(actionValue, gradients);

          // Accumulate gradients.
          totalGradients += gradients;
        }

        // Clamp the accumulated gradients.
        totalGradients.transform(
            [&](double gradient)
            { return std::min(std::max(gradient, -config.GradientLimit()),
            config.GradientLimit()); });

        // Perform async update of the global network.
        updater.Update(learningNetwork.Parameters(),
            config.StepSize(), totalGradients);

        // Sync the local network with the global network.
        network = learningNetwork;

        pendingIndex = 0;
      }

      if (terminal)
        break;
      else
        state = nextState;

      // Update global target network.
      if (totalSteps % config.TargetNetworkSyncInterval() == 0)
      {
        #pragma omp critical
        { targetNetwork = learningNetwork; }
      }

      policy.Anneal();
    }
    return totalReturn;
  }
};

} // namespace rl
} // namespace mlpack

#endif
