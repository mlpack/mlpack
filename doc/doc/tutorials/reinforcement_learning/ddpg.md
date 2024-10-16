# DDPG in mlpack

Deep Deterministic Policy Gradient (DDPG) is a powerful reinforcement learning algorithm that combines the strengths of both Q-learning and policy gradients. DDPG is particularly well-suited for solving continuous action space problems, where traditional Q-learning methods may struggle. 

It leverages deep neural networks to approximate both the Q-value and the policy function. This enables DDPG to learn a deterministic policy that maps states to actions directly, making it well-suited for tasks where precise control is necessary. This algorithm has shown impressive results in a variety of continuous control tasks, making it a popular choice in the field of deep reinforcement learning.

In the following example, we will illustrate how to implement and utilize the Deep Deterministic Policy Gradient (DDPG) algorithm using mlpack for reinforcement learning. We will cover the essential components step by step.

### Setting Up the Environment and Configuration
To begin, we set up the environment and configuration required for the DDPG algorithm. This includes initializing the replay method, which stores and manages experiences for training.

```c++
// Set up the replay method.
RandomReplay<Pendulum> replayMethod(32, 10000);

// Set up the training configuration.
TrainingConfig config;
config.StepSize() = 0.01;
config.TargetNetworkSyncInterval() = 1;
config.UpdateInterval() = 3;
```

### Creating the Actor and Critic Networks
The next step involves creating the neural networks for both the actor and critic. The actor network is responsible for generating actions, while the critic network evaluates the quality of those actions.

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

### Noise
Noise plays a vital role in reinforcement learning algorithms, adding exploration to the agent's actions and enabling the discovery of new states. The Deep Deterministic Policy Gradient (DDPG) algorithm, implemented in mlpack, expects a noise object (of type `NoiseType`) to introduce exploration during the learning process.

For instance, you can use the `GaussianNoise` class to generate uncorrelated noise with a Gaussian distribution, which is particularly useful for continuous action spaces. Here's an example of setting up and creating a `GaussianNoise` instance:

```c++
// Set up the GaussianNoise parameters.
int size = 1; 
double mu = 0.0; 
double sigma = 0.1; 

// Create an instance of the GaussianNoise class.
GaussianNoise gaussianNoise(size, mu, sigma);
```

Alternatively, you can utilize the Ornstein-Uhlenbeck noise process for exploration by creating an instance of the `OUNoise` class. This process generates temporally correlated noise, making it suitable for tasks with inertia or momentum. Here's an example of setting up and creating an `OUNoise` instance:

```c++
// Set up the OUNoise parameters.
int size = 4;
double mu = 0.0;
double theta = 1.0;  
double sigma = 0.1; 

// Create an instance of the OUNoise class.
OUNoise ouNoise(size, mu, theta, sigma);
```

### Configuring and Instantiating the DDPG Agent
With all the necessary components in place, we can now create the DDPG agent. It requires the actor and critic networks, the noise type (in this case, `OUNoise`), and the previously configured settings.

```c++
// Set up Deep Deterministic Policy Gradient agent.
DDPG<Pendulum, decltype(qNetwork), decltype(policyNetwork), 
    OUNoise, AdamUpdate>
    agent(config, qNetwork, policyNetwork, ouNoise, replayMethod);
```

### Training and Convergence
The final part of the code involves training the DDPG agent and monitoring its convergence. We iteratively run episodes, collect returns, and assess the agent's performance. If the agent successfully converges, a deterministic test is performed to validate its behavior.

```c++
// Set the thresholds for testing.
const double rewardThreshold = -400;
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
```

Here is the complete file that you can directly run on your machine. Feel free to experiment with different hyperparameters and noise types to observe their effects on the DDPG algorithm.

```c++
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;

int main()
{
  // Set up the replay method.
  RandomReplay<Pendulum> replayMethod(32, 10000);

  // Set up the training configuration.
  TrainingConfig config;
  config.StepSize() = 0.01;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;

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

  // Set up the OUNoise parameters.
  int size = 1; 
  double mu = 0.0;
  double theta = 1.0; 
  double sigma = 0.1;  

  // Create an instance of the OUNoise class.
  OUNoise ouNoise(size, mu, theta, sigma);

  // Set up Deep Deterministic Policy Gradient agent.
  DDPG<Pendulum, decltype(qNetwork), decltype(policyNetwork), 
      OUNoise, AdamUpdate>
      agent(config, qNetwork, policyNetwork, ouNoise, replayMethod);

  // Set the thresholds for testing.
  const double rewardThreshold = -400;
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
