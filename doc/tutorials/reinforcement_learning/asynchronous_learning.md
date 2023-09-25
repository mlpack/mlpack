# Asynchronous Learning

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
  return false;
};
```

This will train three different agents on three CPU threads asynchronously and
use this data to update the action value estimate.

Voila, that's all there is to it.

Here is the full code to try this right away:

```c++
#include <mlpack.hpp>

using namespace mlpack;

int main()
{
  // Set up the network.
  FFN<MeanSquaredError, GaussianInitialization> model(MeanSquaredError(), GaussianInitialization(0, 0.001));
  model.Add<Linear>(128);
  model.Add<ReLU>();
  model.Add<Linear>(128);
  model.Add<ReLU>();
  model.Add<Linear>(2);

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
    return false;
  };

  for (int i = 0; i < 100; i++)
  {
    agent.Train(measure);
  }
}
```
