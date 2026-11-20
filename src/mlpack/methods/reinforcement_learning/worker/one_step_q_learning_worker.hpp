/**
 * @file methods/reinforcement_learning/worker/one_step_q_learning_worker.hpp
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

#include <ensmallen.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

namespace mlpack {

/**
 * One step Q-Learning worker.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy. *
 */
template <
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
class OneStepQLearningWorker
{
 public:
  using StateType = typename EnvironmentType::State;
  using ActionType = typename EnvironmentType::Action;
  using TransitionType = std::tuple<StateType, ActionType, double, StateType>;

  /**
   * Construct one step Q-Learning worker with the given parameters and
   * environment.
   *
   * @param updater The optimizer.
   * @param environment The reinforcement learning task.
   * @param config Hyper-parameters.
   * @param deterministic Whether it should be deterministic.
   */
  OneStepQLearningWorker(
      const UpdaterType& updater,
      const EnvironmentType& environment,
      const TrainingConfig& config,
      bool deterministic):
      updater(updater),
      #if ENS_VERSION_MAJOR >= 2
      updatePolicy(NULL),
      #endif
      environment(environment),
      config(config),
      deterministic(deterministic),
      pending(config.UpdateInterval())
  { Reset(); }

  /**
   * Copy another OneStepQLearningWorker.
   *
   * @param other OneStepQLearningWorker to copy.
   */
  OneStepQLearningWorker(const OneStepQLearningWorker& other) :
      updater(other.updater),
      #if ENS_VERSION_MAJOR >= 2
      updatePolicy(NULL),
      #endif
      environment(other.environment),
      config(other.config),
      deterministic(other.deterministic),
      steps(other.steps),
      episodeReturn(other.episodeReturn),
      pending(other.pending),
      pendingIndex(other.pendingIndex),
      network(other.network),
      state(other.state)
  {
    #if ENS_VERSION_MAJOR >= 2
    updatePolicy = new typename UpdaterType::template
        Policy<arma::mat, arma::mat>(updater,
                                     network.Parameters().n_rows,
                                     network.Parameters().n_cols);
    #endif

    Reset();
  }

  /**
   * Take ownership of another OneStepQLearningWorker.
   *
   * @param other OneStepQLearningWorker to take ownership of.
   */
  OneStepQLearningWorker(OneStepQLearningWorker&& other) :
      updater(std::move(other.updater)),
      #if ENS_VERSION_MAJOR >= 2
      updatePolicy(NULL),
      #endif
      environment(std::move(other.environment)),
      config(std::move(other.config)),
      deterministic(std::move(other.deterministic)),
      steps(std::move(other.steps)),
      episodeReturn(std::move(other.episodeReturn)),
      pending(std::move(other.pending)),
      pendingIndex(std::move(other.pendingIndex)),
      network(std::move(other.network)),
      state(std::move(other.state))
  {
    #if ENS_VERSION_MAJOR >= 2
    other.updatePolicy = NULL;

    updatePolicy = new typename UpdaterType::template
        Policy<arma::mat, arma::mat>(updater,
                                     network.Parameters().n_rows,
                                     network.Parameters().n_cols);
    #endif
  }

  /**
   * Copy another OneStepQLearningWorker.
   *
   * @param other OneStepQLearningWorker to copy.
   */
  OneStepQLearningWorker& operator=(const OneStepQLearningWorker& other)
  {
    if (&other == this)
      return *this;

    #if ENS_VERSION_MAJOR >= 2
    delete updatePolicy;
    #endif

    updater = other.updater;
    environment = other.environment;
    config = other.config;
    deterministic = other.deterministic;
    steps = other.steps;
    episodeReturn = other.episodeReturn;
    pending = other.pending;
    pendingIndex = other.pendingIndex;
    network = other.network;
    state = other.state;

    #if ENS_VERSION_MAJOR >= 2
    updatePolicy = new typename UpdaterType::template
        Policy<arma::mat, arma::mat>(updater,
                                     network.Parameters().n_rows,
                                     network.Parameters().n_cols);
    #endif

    Reset();

    return *this;
  }

  /**
   * Take ownership of another OneStepQLearningWorker.
   *
   * @param other OneStepQLearningWorker to take ownership of.
   */
  OneStepQLearningWorker& operator=(OneStepQLearningWorker&& other)
  {
    if (&other == this)
      return *this;

    #if ENS_VERSION_MAJOR >= 2
    delete updatePolicy;
    #endif

    updater = std::move(other.updater);
    environment = std::move(other.environment);
    config = std::move(other.config);
    deterministic = std::move(other.deterministic);
    steps = std::move(other.steps);
    episodeReturn = std::move(other.episodeReturn);
    pending = std::move(other.pending);
    pendingIndex = std::move(other.pendingIndex);
    network = std::move(other.network);
    state = std::move(other.state);

    #if ENS_VERSION_MAJOR >= 2
    other.updatePolicy = NULL;

    updatePolicy = new typename UpdaterType::template
        Policy<arma::mat, arma::mat>(updater,
                                     network.Parameters().n_rows,
                                     network.Parameters().n_cols);
    #endif

    return *this;
  }

  /**
   * Clean memory.
   */
  ~OneStepQLearningWorker()
  {
    #if ENS_VERSION_MAJOR >= 2
    delete updatePolicy;
    #endif
  }

  /**
   * Initialize the worker.
   * @param learningNetwork The shared network.
   */
  void Initialize(NetworkType& learningNetwork)
  {
    #if ENS_VERSION_MAJOR == 1
    updater.Initialize(learningNetwork.Parameters().n_rows,
                       learningNetwork.Parameters().n_cols);
    #else
    delete updatePolicy;

    updatePolicy = new typename UpdaterType::template
        Policy<arma::mat, arma::mat>(updater,
                                     learningNetwork.Parameters().n_rows,
                                     learningNetwork.Parameters().n_cols);
    #endif

    // Build local network.
    network = learningNetwork;
  }

  /**
   * The agent will execute one step.
   *
   * @param learningNetwork The shared learning network.
   * @param targetNetwork The shared target network.
   * @param totalSteps The shared counter for total steps.
   * @param policy The shared behavior policy.
   * @param totalReward This will be the episode return if the episode ends
   *     after this step. Otherwise this is invalid.
   * @return Indicate whether current episode ends after this step.
   */
  bool Step(NetworkType& learningNetwork,
            NetworkType& targetNetwork,
            size_t& totalSteps,
            PolicyType& policy,
            double& totalReward)
  {
    // Interact with the environment.
    arma::colvec actionValue;
    network.Predict(state.Encode(), actionValue);
    ActionType action = policy.Sample(actionValue, deterministic);
    StateType nextState;
    double reward = environment.Sample(state, action, nextState);
    bool terminal = environment.IsTerminal(nextState);

    episodeReturn += reward;
    steps++;

    terminal = terminal || steps >= config.StepLimit();
    if (deterministic)
    {
      if (terminal)
      {
        totalReward = episodeReturn;
        Reset();
        // Sync with latest learning network.
        network = learningNetwork;
        return true;
      }
      state = nextState;
      return false;
    }

    #pragma omp atomic
    totalSteps++;

    pending[pendingIndex] = std::make_tuple(state, action, reward, nextState);
    pendingIndex++;

    if (terminal || pendingIndex >= config.UpdateInterval())
    {
      // Initialize the gradient storage.
      arma::mat totalGradients(learningNetwork.Parameters().n_rows,
          learningNetwork.Parameters().n_cols);
      for (size_t i = 0; i < pending.size(); ++i)
      {
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
        arma::mat input = std::get<0>(transition).Encode();
        network.Forward(input, actionValue);
        actionValue[std::get<1>(transition).action] = targetActionValue;

        // Compute gradient.
        arma::mat gradients;
        network.Backward(input, actionValue, gradients);

        // Accumulate gradients.
        totalGradients += gradients;
      }

      // Clamp the accumulated gradients.
      totalGradients.transform(
          [&](double gradient)
          { return std::min(std::max(gradient, -config.GradientLimit()),
          config.GradientLimit()); });

      // Perform async update of the global network.
      #if ENS_VERSION_MAJOR == 1
      updater.Update(learningNetwork.Parameters(), config.StepSize(),
          totalGradients);
      #else
      updatePolicy->Update(learningNetwork.Parameters(),
          config.StepSize(), totalGradients);
      #endif

      // Sync the local network with the global network.
      network = learningNetwork;

      pendingIndex = 0;
    }

    // Update global target network.
    if (totalSteps % config.TargetNetworkSyncInterval() == 0)
    {
      #pragma omp critical
      { targetNetwork = learningNetwork; }
    }

    policy.Anneal();

    if (terminal)
    {
      totalReward = episodeReturn;
      Reset();
      return true;
    }
    state = nextState;
    return false;
  }

 private:
  /**
   * Reset the worker for a new episode.
   */
  void Reset()
  {
    steps = 0;
    episodeReturn = 0;
    pendingIndex = 0;
    state = environment.InitialSample();
  }

  //! Locally-stored optimizer.
  UpdaterType updater;
  #if ENS_VERSION_MAJOR >= 2
  typename UpdaterType::template Policy<arma::mat, arma::mat>* updatePolicy;
  #endif

  //! Locally-stored task.
  EnvironmentType environment;

  //! Locally-stored hyper-parameters.
  TrainingConfig config;

  //! Whether this episode is deterministic or not.
  bool deterministic;

  //! Total steps in current episode.
  size_t steps;

  //! Total reward in current episode.
  double episodeReturn;

  //! Buffer for delayed update.
  std::vector<TransitionType> pending;

  //! Current position of the buffer.
  size_t pendingIndex;

  //! Local network of the worker.
  NetworkType network;

  //! Current state of the agent.
  StateType state;
};

} // namespace mlpack

#endif
