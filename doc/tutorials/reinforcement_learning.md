# Reinforcement Learning Tutorial

Reinforcement Learning is one of the hottest topics right now, with interest
surging after DeepMind published their article on training deep neural networks
to play Atari games to great success. mlpack implements a complete end-to-end
framework for Reinforcement Learning, featuring multiple environments, policies
and methods. Of course, custom environments and policies can be used and plugged
into the existing framework with no runtime overhead.

mlpack implements typical benchmark environments (Acrobot, Mountain car etc.),
commonly used policies, replay methods and supports asynchronous learning as
well. In addition, it can [communicate](https://github.com/zoq/gym_tcp_api) with
the OpenAI Gym toolkit for more environments.

## Reinforcement Learning Environments

mlpack implements a number of the most popular environments used for testing RL
agents and algorithms. These include the Cart Pole, Acrobot, Mountain Car and
their variations. Of course, as mentioned above, you can communicate with OpenAI
Gym for other environments, like the Atari video games.

A key component of mlpack is its extensibility. It is a simple process to create
your own custom environments, specific to your needs, and use it with mlpack's
RL framework. All the environments implement a few specific methods and classes
which are used by the agents while learning.

- `State`: The `State` class is a representation of the environment. For the
  `CartPole`, this would involve storing the position, velocity, angle and
   angular velocity.

- `Action`: For discrete environments, `Action` is a class with an enum naming
  all the possible actions the agent can take in the environment. Continuing
  with the `CartPole` example, the enum would simply contain the two possible
  actions, `backward` and `forward`. For continuous environments, the `Action`
  class contains an array with its size depending on the action space.

- `Sample`: This method is perhaps the heart of the environment, providing
  rewards to the agent depending on the state and the action taken, and updates
  the state based on the action taken as well.

Of course, your custom environment will most likely make use of a number of
helper methods, depending on your application, such as the `Dsdt` method in the
`Acrobot` environment, used in the `RK4` iterative method (also another helper
method) to estimate the next state.

## Components of an RL Agent

A Reinforcement Learning agent, in general, takes actions in an environment in
order to maximize a cumulative reward. To that end, it requires a way to choose
actions (*policy*) and a way to sample previous experiences (*replay*).

An example of a simple policy would be an epsilon-greedy policy. Using such a
policy, the agent will choose actions greedily with some probability epsilon.
This probability is slowly decreased over time, balancing the line between
exploration and exploitation.

Similarly, an example of a simple replay would be a random replay. At each time
step, the interactions between the agent and the environment are saved to a
memory buffer and previous experiences are sampled from the buffer to train the
agent.

Instantiating the components of an agent can be easily done by passing the
Environment as a templated argument and the parameters of the policy/replay to
the constructor.

To create a Greedy Policy and Prioritized Replay for the `CartPole` environment,
we would do the following:

```c++
GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
PrioritizedReplay<CartPole> replayMethod(10, 10000, 0.6);
```

The arguments to `policy` are the initial epsilon values, the interval of
decrease in its value and the value at which epsilon bottoms out and won't be
reduced further. The arguments to `replayMethod` are size of the batch returned,
the number of examples stored in memory, and the degree of prioritization.

In addition to the above components, an RL agent requires many hyperparameters
to be tuned during it's training period. These parameters include everything
from the discount rate of the future reward to whether Double Q-learning should
be used or not. The `TrainingConfig` class can be instantiated and configured as
follows:

```c++
TrainingConfig config;
config.StepSize() = 0.01;
config.Discount() = 0.9;
config.TargetNetworkSyncInterval() = 100;
config.ExplorationSteps() = 100;
config.DoubleQLearning() = false;
config.StepLimit() = 200;
```

The object `config` describes an RL agent, using a step size of 0.01 for the
optimization process, a discount factor of 0.9, sync interval of 200 episodes.
This agent only starts learning after storing 100 exploration steps, has a step
limit of 200, and does not utilize double q-learning.

In this way, we can easily configure an RL agent with the desired
hyperparameters.

## Reinforcement Learning Agents

mlpack provides several powerful reinforcement learning agents that can be used to solve a wide range of reinforcement learning tasks. Below are the key reinforcement learning agents available in mlpack:

- [Q-Learning](reinforcement_learning/q_learning.md)
- [Asynchronous Learning](reinforcement_learning/asynchronous_learning.md)
- [Deep Deterministic Policy Gradient (DDPG)](reinforcement_learning/ddpg.md)
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](reinforcement_learning/td3.md)
- [Soft Actor-Critic (SAC)](reinforcement_learning/sac.md)

Each of these reinforcement learning agents is designed to be highly customizable and flexible, allowing users to easily apply them to various environments and tasks. 

## Further Documentation

For further documentation on the reinforcement learning classes, consult the
documentation in the source code, found in
`mlpack/methods/reinforcement_learning/`.
