# Q-Learning in mlpack

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
  FFN<MeanSquaredError, GaussianInitialization> network(MeanSquaredError(),
      GaussianInitialization(0, 0.001));
  network.Add<Linear>(128);
  network.Add<ReLU>();
  network.Add<Linear>(128);
  network.Add<ReLU>();
  network.Add<Linear>(2);

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
