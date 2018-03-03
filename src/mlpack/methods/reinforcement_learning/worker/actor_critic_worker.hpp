/**
 * @file actor_critic_worker.hpp
 * @author Shangtong Zhang
 *
 * This file is the definition of ActorCriticWorker class,
 * which implements an episode for async advantage actor critic method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_WORKER_ACTOR_CRITIC_WORKER_HPP
#define MLPACK_METHODS_RL_WORKER_ACTOR_CRITIC_WORKER_HPP

#include <mlpack/methods/reinforcement_learning/training_config.hpp>

namespace mlpack {
namespace rl {

/**
 * Advantage actor critic worker.
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
class ActorCriticWorker
{
 public:
  using StateType = typename EnvironmentType::State;
  using ActionType = typename EnvironmentType::Action;
  using TransitionType = std::tuple<StateType, ActionType, double, StateType>;

  /**
   * Construct advantage actor critic worker with given parameters and
   * environment.
   *
   * @param updater The optimizer.
   * @param environment The reinforcement learning task.
   * @param config Hyper-parameters.
   * @param deterministic Whether it should be deterministic.
   */
  ActorCriticWorker(
      const UpdaterType& updater,
      const EnvironmentType& environment,
      const TrainingConfig& config,
      bool deterministic):
      actorUpdater(updater),
      criticUpdater(updater),
      environment(environment),
      config(config),
      deterministic(deterministic),
      pending(config.UpdateInterval())
  { Reset(); }

  /**
   * Initialize the worker.
   *
   * @param learningNetwork The shared network.
   */
  void Initialize(NetworkType& learningNetwork)
  {
    actorUpdater.Initialize(learningNetwork.Actor().Parameters().n_rows,
        learningNetwork.Actor().Parameters().n_cols);
    criticUpdater.Initialize(learningNetwork.Critic().Parameters().n_rows,
        learningNetwork.Critic().Parameters().n_cols);
    // Build local network.
    network = learningNetwork;
  }

  /**
   * The agent will execute one step.
   *
   * @param learningNetwork The shared learning network.
   * @param targetNetwork Not used.
   * @param totalSteps The shared counter for total steps.
   * @param policy The shared behavior policy.
   * @param totalReward This will be the episode return if the episode ends
   *     after this step. Otherwise this is invalid.
   * @return Indicate whether current episode ends after this step.
   */
  bool Step(NetworkType& learningNetwork,
            NetworkType&,
            size_t& totalSteps,
            PolicyType& policy,
            double& totalReward)
  {
    // Interact with the environment.
    arma::colvec actionProb;
    network.Actor().Predict(state.Encode(), actionProb);
    ActionType action = policy.Sample(actionProb, deterministic);
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
      arma::mat totalActorGradients(learningNetwork.Actor().Parameters().n_rows,
          learningNetwork.Actor().Parameters().n_cols, arma::fill::zeros);
      arma::mat totalCriticGradients(
          learningNetwork.Critic().Parameters().n_rows,
          learningNetwork.Critic().Parameters().n_cols, arma::fill::zeros);

      // Bootstrap from the value of next state.
      arma::colvec target(1, arma::fill::zeros);
      if (!terminal)
        network.Critic().Predict(nextState.Encode(), target);

      // Update in reverse order.
      for (int i = pending.size() - 1; i >= 0; --i)
      {
        TransitionType &transition = pending[i];
        target = config.Discount() * target + std::get<2>(transition);

        arma::colvec stateValue;
        network.Critic().Forward(std::get<0>(transition).Encode(), stateValue);
        arma::mat criticGradients;
        network.Critic().Backward(target, criticGradients);
        totalCriticGradients += criticGradients;

        network.Actor().Forward(std::get<0>(transition).Encode(), actionProb);
        // Store the advantage in an one-hot form.
        arma::colvec advantage(actionProb.n_elem, arma::fill::zeros);
        advantage[std::get<1>(transition)] =
            arma::as_scalar(target - stateValue);

        arma::mat actorGradients;
        network.Actor().Backward(advantage, actorGradients);
        totalActorGradients += actorGradients;
      }

      // Clamp the accumulated gradients.
      auto clamp = [&](double gradient)
      {
        return std::min(std::max(gradient, -config.GradientLimit()),
            config.GradientLimit());
      };

      totalActorGradients.transform(clamp);
      totalCriticGradients.transform(clamp);

      actorUpdater.Update(learningNetwork.Actor().Parameters(),
          config.ActorStepSize(), totalActorGradients);
      criticUpdater.Update(learningNetwork.Critic().Parameters(),
          config.CriticStepSize(), totalCriticGradients);

      // Sync the local network with the global network.
      network = learningNetwork;

      pendingIndex = 0;
    }

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
   * Reset the worker for a new episdoe.
   */
  void Reset()
  {
    steps = 0;
    episodeReturn = 0;
    pendingIndex = 0;
    state = environment.InitialSample();
  }

  //! Locally-stored optimizer for the actor network.
  UpdaterType actorUpdater;

  //! Locally-stored optimizer for the critic network.
  UpdaterType criticUpdater;

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

} // namespace rl
} // namespace mlpack

#endif
