#ifndef MLPACK_METHODS_RL_WORKER_ONE_STEP_Q_LEARNING_WORKER_HPP
#define MLPACK_METHODS_RL_WORKER_ONE_STEP_Q_LEARNING_WORKER_HPP

#include <mlpack/methods/reinforcement_learning/training_config.hpp>

namespace mlpack {
namespace rl {

class OneStepQLearningWorker
{
public:
  template<
          typename EnvironmentType,
          typename NetworkType,
          typename UpdaterType,
          typename PolicyType
  >
  static double
  Episode(NetworkType& learningNetwork,
          NetworkType& targetNetwork,
          bool& stop,
          size_t& totalSteps,
          PolicyType& policy,
          UpdaterType& updater,
          EnvironmentType& environment,
          const TrainingConfig& config,
          bool deterministic
          ) {
    using StateType = typename EnvironmentType::State;
    using ActionType = typename EnvironmentType::Action;
    using TransitionType = std::tuple<StateType, ActionType, double, StateType>;

    updater.Initialize(learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols);
    NetworkType network = learningNetwork;
    StateType state = environment.InitialSample();
    size_t steps = 0;
    double totalReturn = 0.0;
    std::vector<TransitionType> pending;
    while (!stop && (!config.StepLimit() || steps < config.StepLimit())) {
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
        else
        {
          state = nextState;
          continue;
        }
      }

#pragma omp atomic
      totalSteps++;

      pending.push_back(std::make_tuple(state, action, reward, nextState));
      if (terminal || pending.size() >= config.UpdateInterval()) {
        arma::mat totalGradients(learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols);
        for (size_t i = 0; i < pending.size(); ++i) {
          TransitionType &transition = pending[i];
          arma::colvec actionValue;

#pragma omp critical
          { targetNetwork.Predict(std::get<3>(transition).Encode(), actionValue); };

          double targetActionValue = actionValue.max();
          if (terminal && i == pending.size() - 1)
            targetActionValue = 0;
          targetActionValue = std::get<2>(transition) + config.Discount() * targetActionValue;
          network.Forward(std::get<0>(transition).Encode(), actionValue);
          actionValue[std::get<1>(transition)] = targetActionValue;
          arma::mat gradients;
          network.Backward(actionValue, gradients);
          totalGradients += gradients;
        }
        totalGradients = arma::clamp(totalGradients, -config.GradientLimit(), config.GradientLimit());
        updater.Update(learningNetwork.Parameters(), config.StepSize(), totalGradients);
        network = learningNetwork;
        pending.clear();
      }

      if (terminal)
        break;
      else
        state = nextState;

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



}
}

#endif
