/**
 * @file one_step_q_learning_worker.hpp
 * @author Shangtong Zhang
 * @author Arsen Zahray
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

#include <mlpack/methods/reinforcement_learning/worker/worker_base.hpp>
#include <mlpack/methods/reinforcement_learning/worker/q_learning_worker_transition_type.hpp>

namespace mlpack {
namespace rl {

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
class OneStepQLearningWorker:public WorkerBase< EnvironmentType, NetworkType, UpdaterType, PolicyType, QLearningWorkerTransitionType<typename EnvironmentType::State,typename EnvironmentType::Action>>
{
    using base = WorkerBase< EnvironmentType, NetworkType, UpdaterType, PolicyType, QLearningWorkerTransitionType<typename EnvironmentType::State,typename EnvironmentType::Action>>;
    //using qWorkerType = QLearningWorkerTransitionType<EnvironmentType>;
 public:
     using StateType = typename EnvironmentType::State;
     using ActionType = typename EnvironmentType::Action;

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
        bool deterministic)
		: base(updater,environment,config,deterministic)
    { }

    /**
     * Copy another OneStepQLearningWorker.
     *
     * @param other OneStepQLearningWorker to copy.
     */
    OneStepQLearningWorker(const OneStepQLearningWorker& other) :
        base(std::forward<const OneStepQLearningWorker&>(other))
    {}

    /**
     * Take ownership of another OneStepQLearningWorker.
     *
     * @param other OneStepQLearningWorker to take ownership of.
     */
    OneStepQLearningWorker(OneStepQLearningWorker&& other) :
        base(std::forward<OneStepQLearningWorker&&>(other))
    {}

    /**
     * Copy another OneStepQLearningWorker.
     *
     * @param other OneStepQLearningWorker to copy.
     */
    OneStepQLearningWorker& operator=(const OneStepQLearningWorker& other)
    {
        if (&other == this)
            return *this;

        base::operator=(std::forward(other));

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

        base::operator=(std::forward(other));

        return *this;
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
            double& totalReward) override
  {
    // Interact with the environment.
    arma::colvec actionValue;
    this->network.Predict(this->state.Encode(), actionValue);
    ActionType action = policy.Sample(actionValue, this->deterministic);
    StateType nextState;
    double reward = this->environment.Sample(this->state, action, nextState);
    bool terminal = this->environment.IsTerminal(nextState);

    this->episodeReturn += reward;
    this->steps++;

    terminal = terminal || this->steps >= this->config.StepLimit();
    if (this->deterministic)
    {
      if (terminal)
      {
        totalReward = this->episodeReturn;
        this->Reset();
        // Sync with latest learning network.
        this->network = learningNetwork;
        return true;
      }
      this->state = nextState;
      return false;
    }

    #pragma omp atomic
    totalSteps++;

    this->pending[this->pendingIndex] = { this->state, action, reward, nextState };
    this->pendingIndex++;

    if (terminal || this->pendingIndex >= this->config.UpdateInterval())
    {
      // Initialize the gradient storage.
      arma::mat totalGradients(learningNetwork.Parameters().n_rows,
          learningNetwork.Parameters().n_cols, arma::fill::zeros);
      for (size_t i = 0; i < this->pending.size(); ++i)
      {
        auto &transition = this->pending[i];

        // Compute the target state-action value.
        arma::colvec actionValue;
        #pragma omp critical
        {
          targetNetwork.Predict(
              transition.nextState.Encode(), actionValue);
        };
        double targetActionValue = actionValue.max();
        if (terminal && i == this->pending.size() - 1)
        {
            targetActionValue = 0;
        }
        targetActionValue = transition.reward +
            this->config.Discount() * targetActionValue;

        // Compute the training target for current state.
        auto input=transition.state.Encode();
        this->network.Forward(input, actionValue);
        actionValue[transition.action] = targetActionValue;

        // Compute gradient.
        arma::mat gradients;
        //this->network.Backward(actionValue, gradients);
        this->network.Backward(input, actionValue, gradients);

        // Accumulate gradients.
        totalGradients += gradients;
      }

      // Clamp the accumulated gradients.
      totalGradients.transform(
          [&](double gradient)
          { 
		  	return std::min(std::max(gradient, -this->config.GradientLimit()),
              this->config.GradientLimit()); 
	      });

      // Perform async update of the global network.
      #if ENS_VERSION_MAJOR == 1
      this->updater.Update(learningNetwork.Parameters(), this->config.StepSize(),
          totalGradients);
      #else
      this->updatePolicy->Update(learningNetwork.Parameters(),
          this->config.StepSize(), totalGradients);
      #endif

      // Sync the local network with the global network.
      this->network = learningNetwork;

      this->pendingIndex = 0;
    }

    // Update global target network.
    if (totalSteps % this->config.TargetNetworkSyncInterval() == 0)
    {
      #pragma omp critical
      { targetNetwork = learningNetwork; }
    }

    policy.Anneal();

    if (terminal)
    {
      totalReward = this->episodeReturn;
      this->Reset();
      return true;
    }
    this->state = nextState;
    return false;
  }

};

} // namespace rl
} // namespace mlpack

#endif
