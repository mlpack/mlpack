#ifndef MLPACK_METHODS_RL_WORKER_ONE_STEP_Q_LEARNING_HPP
#define MLPACK_METHODS_RL_WORKER_ONE_STEP_Q_LEARNING_HPP

namespace mlpack {
namespace rl {

template<
        typename WorkerType,
        typename EnvironmentType,
        typename NetworkType,
        typename UpdaterType,
        typename PolicyType
>
double
AsyncLearning<WorkerType, EnvironmentType, NetworkType, UpdaterType, PolicyType>::
Episode(NetworkType& learningNetwork,
        NetworkType& targetNetwork,
        bool& stop,
        size_t& totalSteps,
        PolicyType& policy,
        bool deterministic,
        OneStepQLearningWorker) {
  updater.Initialize(learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols);
  NetworkType network = learningNetwork;
  using TransitionType = std::tuple<StateType, ActionType, double, StateType>;
  StateType state = environment.InitialSample();
  size_t steps = 0;
  double totalReturn = 0.0;
  std::vector<TransitionType> pending;
  while (!stop && steps < stepLimit) {
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
    if (terminal || pending.size() >= updateInterval) {
      arma::mat totalGradients(learningNetwork.Parameters().n_rows, learningNetwork.Parameters().n_cols);
      for (size_t i = 0; i < pending.size(); ++i) {
        TransitionType &transition = pending[i];
        arma::colvec actionValue;

        #pragma omp critical
        { targetNetwork.Predict(std::get<3>(transition).Encode(), actionValue); };

        double targetActionValue = actionValue.max();
        if (terminal && i == pending.size() - 1)
          targetActionValue = 0;
        targetActionValue = std::get<2>(transition) + discount * targetActionValue;
        network.Forward(std::get<0>(transition).Encode(), actionValue);
        actionValue[std::get<1>(transition)] = targetActionValue;
        arma::mat gradients;
        network.Backward(actionValue, gradients);
        totalGradients += gradients;
      }
      totalGradients = arma::clamp(totalGradients, -gradientLimit, gradientLimit);
      updater.Update(learningNetwork.Parameters(), stepSize, totalGradients);
      network = learningNetwork;
      pending.clear();
    }

    if (terminal)
      break;
    else
      state = nextState;

    if (totalSteps % targetNetworkSyncInterval == 0)
    {
      #pragma omp critical
      { targetNetwork = learningNetwork; }
    }

    policy.Anneal();
  }
  return totalReturn;
}

}
}

#endif
