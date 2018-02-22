

// Collect an entire episode with existing policy.
template<>
void VanillaPolicyGradient<>::collectEpisode()
{
  // Start a new episode.
  episode.clear();
  // Get initial state from environment.
  StateType state = environment.InitialSample();
  // Loop until you don't hit a terminal state taking actions according to your current policy.
  while(true)
  {
  	// Get probability distribution for actions predicted by your current network.
    arma::colvec actionProb;
    policyNetwork.Predict(state.Encode(), actionProb);
    // Get an action from current state according to the predicted probability distribution.
    ActionType action = policy.Sample(actionProb, deterministic);
    StateType nextState;
    // Execute the action, get the next state, and reward.
    double reward = environment.Sample(state, action, nextState);
    bool terminal = environment.IsTerminal(nextState);
    // Add this step to episode.
    episode.push_back(std::make_tuple(state, action, reward, nextState));
    // If we have reached a terminal state, end episode.
    if(terminal) break;
  }
}

// After you have an episode, step through each transition in the episode, and update your policy network at each step.
template<>
void VanillaPolicyGradient<>::Step()
{
  int episode_length = episode.size();
  double value = 0.0;
  cumulative_value.resize(episode_length);

  // Iterate through the transitions in the episode, and compute cumulative values of the states as a summation of discounted future rewards from that state onwards.
  for (int i = episode_length-1; i >= 0; i--)
  {
  	TransitionType transition = episode[i]; 
    value += std::get<2>(transition);
    cumulative_value[i] = value;
    value *= config.Discount();
  }

  for(int i = 0; i < episode_length; i++)
  {
  	TransitionType transition = episode[i];
  	StateType state = std::get<0>(transition);
  	ActionType action = std::get<1>(transition);
    arma::colvec actionProb;
    // Get the action probabilities predicted by policyNetwork for this state.
    policyNetwork.Forward(state.Encode(), actionProb);
    // Compute advantage vector: it is simply the state value.
    arma::colvec advantage(actionProb.n_elem);
    advantage.fill(cumulative_value[i]);
    int actionIndex = static_cast<int>(action);
    // Doing this because gradient with respect to the action taken at this step is (1-p(a)).
    // For other actions, it is -p(a).
    advantage(actionIndex) = advantage(actionIndex) * (actionProb(actionIndex) - 1) / actionProb(actionIndex);
    arma::mat gradients;
    // Do the backward pass and compute gradients with respect to parameters of the network.
    policyNetwork.Backward(advantage, gradients);
    // Update the policy network parameters.
    policyNetworkUpdater.Update(policyNetwork.Parameters(), config.PolicyNetworkStepSize(), gradients);
  }
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
  EnvironmentTYpe environment;

  //! SamplePolicy to pick action as per policy network.
  SamplePolicy<> policy;


  