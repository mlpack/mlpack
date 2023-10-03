# SAC in mlpack

Soft Actor-Critic (SAC) stands as an advanced reinforcement learning algorithm that pushes the boundaries of learning capabilities beyond previous methods. Like its predecessors, SAC is tailored to handle challenges in continuous action space problems, where conventional Q-learning approaches might fall short.

SAC introduces innovative strategies that contribute to improved stability and effectiveness throughout the learning journey. A notable enhancement is the utilization of soft Q-networks, addressing the overestimation bias inherent in traditional Q-learning algorithms. Furthermore, SAC incorporates entropy maximization to encourage exploration, leading to more robust policy learning.

Deep neural networks come into play to approximate Q-values and policy functions, facilitating the learning of accurate and efficient deterministic policies for tasks that demand precise control. SAC has showcased remarkable achievements across diverse continuous control challenges, cementing its position as a leading-edge technique in the realm of deep reinforcement learning.

In the forthcoming example, we will embark on a comprehensive exploration of the Soft Actor-Critic (SAC) algorithm's implementation and utilization within mlpack. Through step-by-step guidance, we will unravel the core components of SAC and showcase its prowess in handling intricate continuous control scenarios.

### Setting Up the Environment and Configuration
Let's initiate the environment and establish essential configurations for the SAC algorithm. This involves setting up the replay mechanism, which plays a pivotal role in managing and preserving experiences vital for the training phase.

```c++
// Set up the replay method.
RandomReplay<Pendulum> replayMethod(32, 10000);

TrainingConfig config;
config.StepSize() = 0.01;
config.TargetNetworkSyncInterval() = 1;
config.UpdateInterval() = 3;
config.Rho() = 0.001;
```

### Creating the Actor and Critic Networks
Moving forward, we advance to the next phase by constructing neural networks for both the actor and critic modules. The actor network assumes a critical role in generating actions, while the critic network undertakes the responsibility of evaluating the effectiveness and significance of these actions.

```c++
FN<EmptyLoss, GaussianInitialization>
    policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
policyNetwork.Add(new Linear(128));
policyNetwork.Add(new ReLU());
policyNetwork.Add(new Linear(1));
policyNetwork.Add(new TanH());

FFN<EmptyLoss, GaussianInitialization>
    qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
qNetwork.Add(new Linear(128));
qNetwork.Add(new ReLU());
qNetwork.Add(new Linear(1));
```

### Configuring and Instantiating the SAC Agent
With all the fundamental components in place, we are now ready to initialize the SAC agent. This entails combining the actor and critic networks and integrating the pre-configured settings to create a cohesive SAC agent instance.

```c++
// Set up Soft actor-critic agent.
SAC<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
    agent(config, qNetwork, policyNetwork, replayMethod);
```

Below is the complete code file that you can execute on your local machine. Feel free to explore various hyperparameters to observe their impact on the SAC algorithm's behavior and performance. 

```c++
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;

int main()
{
  // Set up the replay method.
  RandomReplay<Pendulum> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;
  config.Rho() = 0.001;


  FFN<EmptyLoss, GaussianInitialization>
      policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear(128));
  policyNetwork.Add(new ReLU());
  policyNetwork.Add(new Linear(1));
  policyNetwork.Add(new TanH());

  FFN<EmptyLoss, GaussianInitialization>
      qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear(128));
  qNetwork.Add(new ReLU());
  qNetwork.Add(new Linear(1));

  // Set up Soft actor-critic agent.
  SAC<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  // Set the thresholds for testing.
  const double rewardThreshold = -300;
  const size_t noOfEpisodes = 1000;
  const size_t consecutiveEpisodesTest = 10;

  std::vector<double> returnList;
  size_t episodes = 0;
  while (true)
  {
    double episodeReturn = agent.Episode();
    episodes += 1;
    returnList.push_back(episodeReturn);

    if (returnList.size() > consecutiveEpisodesTest)
      returnList.erase(returnList.begin());

    double averageReturn = std::accumulate(returnList.begin(),
        returnList.end(), 0.0) / returnList.size();

    std::cout << "Average return in last " << returnList.size()
        << " consecutive episodes: " << averageReturn
        << " Episode return: " << episodeReturn << std::endl;

    if (averageReturn > rewardThreshold &&
        returnList.size() >= consecutiveEpisodesTest)
    {
      // Test the agent for 10 episodes.
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      std::cout << "Agent successfully converged.\n"
          << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }

    if (episodes > noOfEpisodes)
    {
      std::cout << "Agent failed to converge." << std::endl;
      break;
    }
  }

  return 0;
}
```