# TD3 in mlpack

Twin Delayed Deep Deterministic Policy Gradient (TD3) is an advanced reinforcement learning algorithm that builds upon the foundation of DDPG. Just like DDPG, TD3 is designed for tackling continuous action space problems, where the conventional Q-learning approaches may face challenges.

TD3 introduces several innovations that enhance the stability and effectiveness of the learning process. One of the key enhancements is the utilization of twin Q-networks to address the overestimation bias inherent in traditional Q-learning algorithms. Additionally, TD3 incorporates a target policy smoothing technique, which further improves the stability of policy updates.

This algorithm leverages deep neural networks to approximate Q-values and policy functions, enabling it to learn precise deterministic policies for tasks that demand accurate control. TD3 has demonstrated remarkable performance across a wide range of continuous control tasks, solidifying its position as a cutting-edge method in deep reinforcement learning.

In the following example, we will delve into the implementation and utilization of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm using mlpack. Through a step-by-step guide, we will explore the fundamental components of TD3 and showcase its potential in solving complex continuous control problems.

### Setting Up the Environment and Configuration
Let's start by preparing the environment and configuring the necessary settings for the TD3 algorithm. This entails the initialization of the replay mechanism, which effectively manages and retains experiences crucial for the training process.

```c++
// Set up the replay method.
RandomReplay<Pendulum> replayMethod(32, 10000);

// Set up the TD3 training configuration.
TrainingConfig config;
config.StepSize() = 0.01;
config.TargetNetworkSyncInterval() = 1;
config.UpdateInterval() = 3;
config.Rho() = 0.001;
```

### Creating the Actor and Critic Networks
Proceeding to the subsequent stage, we proceed with the establishment of neural networks for both the actor and critic components. The actor network plays a pivotal role in producing actions, whereas the critic network is tasked with assessing the efficacy and value of these actions.

```c++
// Set up Actor network.
FFN<EmptyLoss, GaussianInitialization>
    policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
policyNetwork.Add(new Linear(128));
policyNetwork.Add(new ReLU());
policyNetwork.Add(new Linear(1));
policyNetwork.Add(new TanH());

// Set up Critic network.
FFN<EmptyLoss, GaussianInitialization>
    qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
qNetwork.Add(new Linear(128));
qNetwork.Add(new ReLU());
qNetwork.Add(new Linear(1));
```

### Configuring and Instantiating the TD3 Agent
With all the essential elements aligned, we are now poised to instantiate the TD3 agent. This involves assembling the actor and critic networks, as well as integrating the previously configured settings. Unlike DDPG, TD3 does not require a noise type.

```c++
// Set up Twin Delayed Deep Deterministic policy gradient agent.
TD3<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
    agent(config, qNetwork, policyNetwork, replayMethod);
```

Below is the complete code file that you can execute on your local machine. Feel free to explore various hyperparameters to observe their impact on the TD3 algorithm's behavior and performance. 

```c++
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;

int main()
{
  // Set up the replay method.
  RandomReplay<Pendulum> replayMethod(32, 10000);

  // Set up the TD3 training configuration.
  TrainingConfig config;
  config.StepSize() = 0.01;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;
  config.Rho() = 0.001;

  // Set up Actor network.
  FFN<EmptyLoss, GaussianInitialization>
      policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear(128));
  policyNetwork.Add(new ReLU());
  policyNetwork.Add(new Linear(1));
  policyNetwork.Add(new TanH());

  // Set up Critic network.
  FFN<EmptyLoss, GaussianInitialization>
      qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear(128));
  qNetwork.Add(new ReLU());
  qNetwork.Add(new Linear(1));

  // Set up Twin Delayed Deep Deterministic policy gradient agent.
  TD3<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
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