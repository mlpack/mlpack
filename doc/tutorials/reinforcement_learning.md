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

## Q-Learning in mlpack

Here, we demonstrate Q-Learning in mlpack through the use of a simple example,
the training of a Q-Learning agent on the `CartPole` environment. The code has
been broken into chunks for easy understanding.

```c++
#include <mlpack.hpp>

using namespace mlpack;
using namespace ens;
using namespace mlpack::rl;
```

We include all the necessary components of our toy example and declare
namespaces for convenience.

```c++
int main()
{
  // Set up the network.
  SimpleDQN<> model(4, 64, 32, 2);
```

The first step in setting our Q-learning agent is to setup the network for it to
use. `SimpleDQN` class creates a simple feed forward network with 2 hidden
layers.  The network constructed here has an input shape of 4 and output shape
of 2. This corresponds to the structure of the `CartPole` environment, where
each state is represented as a column vector with 4 data members (position,
velocity, angle, angular velocity). Similarly, the output shape is represented
by the number of possible actions, which in this case, is only 2 (`foward` and
`backward`).

We can also use mlpack's ann module to set up a custom `FFN` network. For
example, here we use a single hidden layer. However, the Q-Learning agent
expects the object to have a `ResetNoise` method which `SimpleDQN` has.  We
can't pass mlpack's `FFN` network directly. Instead, we have to wrap it into
`SimpleDQN` object.

```c++
int main()
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> network(MeanSquaredError<>(),
      GaussianInitialization(0, 0.001));
  network.Add<Linear<>>(4, 128);
  network.Add<ReLULayer<>>();
  network.Add<Linear<>>(128, 128);
  network.Add<ReLULayer<>>();
  network.Add<Linear<>>(128, 2);

  SimpleDQN<> model(network);

```

The next step would be to setup the other components of the Q-learning agent,
namely its policy, replay method and hyperparameters.

```c++
 // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<CartPole> replayMethod(10, 10000);

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.9;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;
```

And now, we get to the heart of the program, declaring a Q-Learning agent.

```c++
  QLearning<CartPole, decltype(model), AdamUpdate, decltype(policy)>
      agent(config, model, policy, replayMethod);
```

Here, we call the `QLearning` constructor, passing in the type of environment,
network, updater, policy and replay. We use `decltype(var)` as a shorthand for
the variable, saving us the trouble of copying the lengthy templated type.

We pass references of the objects we created, as parameters to `QLearning`
class.

Now, we have our Q-Learning agent `agent` ready to be trained on the Cart Pole
environment.

```c++
  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > 1000)
    {
      std::cout << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }

    /**
     * Reaching running average return 35 is enough to show it works.
     */
    std::cout << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 35)
      break;
  }
  if (converged)
    std::cout << "Hooray! Q-Learning agent successfully trained" << std::endl;

  return 0;
}
```

We set up a loop to train the agent. The exit condition is determined by the
average reward which can be computed with `arma::running_stat`. It is used for
storing running statistics of scalars, which in this case is the reward signal.
The agent can be said to have converged when the average return reaches a
predetermined value (i.e. > 35).

Conversely, if the average return does not go beyond that amount even after a
thousand episodes, we can conclude that the agent will not converge and exit the
training loop.

## Asynchronous Learning

In 2016, Researchers at Deepmind and University of Montreal published their
paper "Asynchronous Methods for Deep Reinforcement Learning". In it they
described asynchronous variants of four standard reinforcement learning
algorithms:

 - One-Step SARSA
 - One-Step Q-Learning
 - N-Step Q-Learning
 - Advantage Actor-Critic(A3C)

Online RL algorithms and Deep Neural Networks make an unstable combination
because of the non-stationary and correlated nature of online updates. Although
this is solved by Experience Replay, it has several drawbacks: it uses more
memory and computation per real interaction; and it requires off-policy learning
algorithms.

Asynchronous methods, instead of experience replay, asynchronously executes
multiple agents in parallel, on multiple instances of the environment, which
solves all the above problems.

Here, we demonstrate Asynchronous Learning methods in mlpack through the
training of an async agent. Asynchronous learning involves training several
agents simultaneously. Here, each of the agents are referred to as "workers".
Currently mlpack has One-Step Q-Learning worker, N-Step Q-Learning worker and
One-Step SARSA worker.

Let's examine the sample code in chunks.

Here we don't use experience replay, and instead of a single policy, we use
three different policies, each corresponding to its worker. Number of workers
created, depends on the number of policies given in the Aggregated Policy. The
column vector contains the probability distribution for each child policy. We
should make sure its size is same as the number of policies and the sum of its
elements is equal to 1.

```
AggregatedPolicy<GreedyPolicy<CartPole>> policy({GreedyPolicy<CartPole>(0.7, 5000, 0.1),
                                                 GreedyPolicy<CartPole>(0.7, 5000, 0.01),
                                                 GreedyPolicy<CartPole>(0.7, 5000, 0.5)},
                                                 arma::colvec("0.4 0.3 0.3"));
```

Now, we will create the `OneStepQLearning` agent. We could have used
`NStepQLearning` or `OneStepSarsa` here according to our requirement.

```c++
OneStepQLearning<CartPole, decltype(model), ens::AdamUpdate, decltype(policy)>
    agent(std::move(config), std::move(model), std::move(policy));
```

Here, unlike the Q-Learning example, instead of the entire while loop, we use
the `Train()` method of the Asynchronous Learning class inside a for loop. 100
training episodes will take around 50 seconds.

```c++
for (int i = 0; i < 100; i++)
{
  agent.Train(measure);
}
```

What is "measure" here? It is a lambda function which returns a boolean value
(indicating the end of training) and accepts the episode return (total reward of
a deterministic test episode) as parameter.  So, let's create that.

```c++
arma::vec returns(20, arma::fill::zeros);
size_t position = 0;
size_t episode = 0;

auto measure = [&returns, &position, &episode](double episodeReturn)
{
  if(episode > 10000) return true;

  returns[position++] = episodeReturn;
  position = position % returns.n_elem;
  episode++;

  std::cout << "Episode No.: " << episode
      << "; Episode Return: " << episodeReturn
      << "; Average Return: " << arma::mean(returns) << std::endl;
};
```

This will train three different agents on three CPU threads asynchronously and
use this data to update the action value estimate.

Voila, that's all there is to it.

Here is the full code to try this right away:

```c++
#include <mlpack.hpp>

using namespace mlpack;
using namespace mlpack::rl;

int main()
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(), GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);

  AggregatedPolicy<GreedyPolicy<CartPole>> policy({GreedyPolicy<CartPole>(0.7, 5000, 0.1),
                                                   GreedyPolicy<CartPole>(0.7, 5000, 0.01),
                                                   GreedyPolicy<CartPole>(0.7, 5000, 0.5)},
                                                   arma::colvec("0.4 0.3 0.3"));

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.9;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;

  OneStepQLearning<CartPole, decltype(model), ens::VanillaUpdate, decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy));

  arma::vec returns(20, arma::fill::zeros);
  size_t position = 0;
  size_t episode = 0;

  auto measure = [&returns, &position, &episode](double episodeReturn)
  {
    if(episode > 10000) return true;

    returns[position++] = episodeReturn;
    position = position % returns.n_elem;
    episode++;

    std::cout << "Episode No.: " << episode
        << "; Episode Return: " << episodeReturn
        << "; Average Return: " << arma::mean(returns) << std::endl;
  };

  for (int i = 0; i < 100; i++)
  {
    agent.Train(measure);
  }
}
```

## Further Documentation

For further documentation on the reinforcement learning classes, consult the
documentation in the source code, found in
`mlpack/methods/reinforcement_learning/`.
