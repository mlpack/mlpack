# Q-Learning in mlpack

This tutorial demonstrates how to use **Q-Learning** in mlpack by training an
agent on the classic CartPole environment.

The example is intentionally simple and broken into sections so each step is
easy to follow.

---

## Prerequisites

Before starting, you should:

* know basic C++
* have mlpack installed
* be able to build programs using CMake

Verify installation with:

```bash
mlpack_knn --help
```

---

## Include headers and namespaces

```c++
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;
using namespace mlpack::rl;
```

We include the required components and bring a few namespaces into scope for
convenience.

---

## Creating a Q-Learning network (SimpleDQN)

```c++
int main()
{
  // Set up the network.
  SimpleDQN<> model(4, 64, 32, 2);
```

The first step is to **set up the neural network** that the Q-Learning agent
will use to approximate the Q-function.

`SimpleDQN` creates a small feed-forward network with two hidden layers.

* Input size: 4
  (cart position, cart velocity, pole angle, angular velocity)
* Output size: 2
  (`forward`, `backward` actions)

The output corresponds to the number of actions available in the environment.

---

## Using a custom FFN network (optional)

You may want to design your own network with mlpack’s ANN module.

```c++
int main()
{
  // Set up a custom network.
  FFN<MeanSquaredError, GaussianInitialization> network(
      MeanSquaredError(), GaussianInitialization(0, 0.001));

  network.Add<Linear>(128);
  network.Add<ReLU>();
  network.Add<Linear>(128);
  network.Add<ReLU>();
  network.Add<Linear>(2);

  // Wrap the network so it provides the required interface.
  SimpleDQN<> model(network);
```

### Why wrap the network?

`FFN` does not implement some functions that Q-Learning requires,
such as `ResetNoise()`.

`SimpleDQN` provides this interface, so we wrap the model before using it.

---

## Configure policy, replay buffer, and hyperparameters

```c++
  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<CartPole> replayMethod(10, 10000);

  // Training configuration.
  TrainingConfig config;
  config.StepSize() = 0.01;                // Learning rate
  config.Discount() = 0.9;                 // Reward discount factor (gamma)
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;
```

* The **policy** controls exploration and exploitation.
* The **replay buffer** stores past transitions for training.
* `TrainingConfig` defines core learning hyperparameters.

---

## Create the Q-Learning agent

```c++
  QLearning<CartPole, decltype(model), AdamUpdate, decltype(policy)>
      agent(config, model, policy, replayMethod);
```

We pass:

* the environment type
* the neural network model
* the optimizer
* the policy
* the replay buffer

`decltype(model)` simply avoids repeating a long templated type.

---

## Training loop and convergence

```c++
  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;

  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes++;

    if (episodes > 1000)
    {
      std::cout << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }

    // Reaching an average return of ~35 shows learning progress.
    std::cout << "Average return: " << averageReturn.mean()
              << " Episode return: " << episodeReturn << std::endl;

    if (averageReturn.mean() > 35)
      break;
  }

  if (converged)
    std::cout << "Q-Learning agent successfully trained." << std::endl;

  return 0;
}
```

We train the agent one episode at a time and track the **running average
return** using `arma::running_stat`.

In this example, we consider training successful when the running average reward
exceeds **35**. This value is a heuristic chosen only for demonstration.

If the agent does not reach this threshold after 1000 episodes, training stops.

---

## Build and run

Save the file (for example as `q_learning.cpp`) and build it:

```bash
mkdir build
cd build
cmake ..
make
./q_learning
```

---

## Summary

In this tutorial, you learned how to:

* construct a neural network for Q-Learning
* configure replay, policy, and training parameters
* initialize a Q-Learning agent
* train and evaluate convergence on CartPole

You can extend this example by:

* experimenting with different architectures
* tuning hyperparameters
* trying other environments
* enabling Double Q-Learning