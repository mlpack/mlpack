#ifndef MLPACK_METHODS_RL_POLICY_GRADIENT_IMPL_HPP
#define MLPACK_METHODS_RL_POLICY_GRADIENT_IMPL_HPP

#include <cstddef>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <armadillo>


namespace mlpack {
namespace rl {

/**
 * Vanilla Policy Gradient implementation.
 *
 * @tparam EnvironmentType The type of the reinforcement learning task.
 * @tparam NetworkType The type of the network model.
 * @tparam UpdaterType The type of the optimizer.
 * @tparam PolicyType The type of the behavior policy.
 */
template<
  typename EnvironmentType,
  typename NetworkType,
  typename UpdaterType,
  typename PolicyType
>
class PolicyGradient
{
public:

  // Convenient typedefs.
  using StateType = typename EnvironmentType::State;
  using ActionType = typename EnvironmentType::Action;
  using TransitionType = std::tuple<StateType, ActionType, double, StateType>;

  /**
   * Construct Vanilla Policy Gradient model with given parameters and
   * environment.
   *
   * @param policyNetwork The network that predicts the policy.
   * @param updater The optimizer.
   * @param environment The reinforcement learning task.
   * @param policy The policy for taking actions.
   * @param config Hyper-parameters.
   */
  PolicyGradient(
    const NetworkType& policyNetwork,
    const UpdaterType& policyNetworkUpdater,
    const EnvironmentType& environment,
    const PolicyType& policy,
    const TrainingConfig& config
    ):
    policyNetwork(policyNetwork),
    policyNetworkUpdater(policyNetworkUpdater),
    environment(environment),
    policy(policy),
    config(config)
  { /* Nothing to do here. */ }
  
  // Collect an entire episode and train the model for the episode step by step.
  void Episode()
  {
    // Start a new episode.
    episode.clear();
    // Get initial state from environment.
    state = environment.InitialSample();
    // Loop until you don't hit a terminal state taking actions according to your current policy.
    while(true)
    {
    	// Get probability distribution for actions predicted by your current network.
      arma::colvec actionProb;
      policyNetwork.Predict(state.Encode(), actionProb);
      // Get an action from current state according to the predicted probability distribution.
      action = policy.Sample(actionProb);
      StateType nextState;
      // Execute the action, get the next state, and reward.
      double reward = environment.Sample(state, action, nextState);
      bool terminal = environment.IsTerminal(nextState);
      // Add this step to episode.
      episode.push_back(std::make_tuple(state, action, reward, nextState));
      // If we have reached a terminal state, end episode.
      if(terminal) break;
    }
  
    int episode_length = episode.size();
    double value = 0.0;
    cumulative_value.resize(episode_length);
  
    // Iterate through the transitions in the episode, and compute cumulative values of the states as a summation of discounted future rewards from that state onwards.
    for (int i = episode_length-1; i >= 0; i--)
    {
    	transition = episode[i]; 
      value += std::get<2>(transition);
      cumulative_value[i] = value;
      value *= config.Discount();
    }
    // Presently, an entire episode has been collected and cumulative_value has got discounted rewards for every state visited during the collected episode.

    // Run through collected episode step by step.
    currentStep = 0;
    for(int i = 0; i < episode_length; i++)
    {
      transition = episode[i];
      state = std::get<0>(transition);
      action = std::get<1>(transition);
      Step();
      currentStep += 1;
    }
  }
  
  // Step through a transition in an episode, and update your policy network.
  void Step()
  {
    arma::colvec actionProb;
    // Get the action probabilities predicted by policyNetwork for this state.
    policyNetwork.Forward(state.Encode(), actionProb);
    // Compute advantage vector: it is simply the state value.
    arma::colvec advantage(actionProb.n_elem);
    advantage.fill(cumulative_value[currentStep]);
    arma::mat gradients;
    int actionIndex = static_cast<int>(action);
    // Doing this because gradient with respect to the action taken at this step is (1-p(a)).
    // For other actions, it is -p(a).
    advantage(actionIndex) = advantage(actionIndex) * (actionProb(actionIndex) - 1) / actionProb(actionIndex);
    // Do the backward pass and compute gradients with respect to parameters of the network.
    policyNetwork.Backward(advantage, gradients);
    // Update the policy network parameters.
    policyNetworkUpdater.Update(policyNetwork.Parameters(), config.StepSize(), gradients);
  }
  
private:
  
  //! An episode, consisting of a list of transitions.
  std::vector<TransitionType> episode;

  //! Discounted rewards at every time step in an episode.
  std::vector<double> cumulative_value;

  //! Locally stored optimizer for the policyNetwork.
  UpdaterType policyNetworkUpdater;

  //! Locally stored hyperparameters.
  TrainingConfig config;

  //! Local policy network.
  NetworkType policyNetwork;

  //! Locally stored task.
  EnvironmentType environment;

  //! Local action.
  ActionType action;

  //! Local state.
  StateType state;

  //! Local transition.
  TransitionType transition;

  //! Policy to pick action as per policy network.
  PolicyType policy;

  //! Variable to maintain visibility of current step acrosse methods.
  int currentStep = 0;
};

} // namespace rl.
} // namespace mlpack.

#endif