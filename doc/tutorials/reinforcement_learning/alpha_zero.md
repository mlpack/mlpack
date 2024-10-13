# AlphaZero
Reinforcement Learning algorithms often face challenges when dealing with large action spaces and sparse rewards. AlphaZero, an extension of Monte Carlo Tree Search (MCTS), overcomes these limitations by using both value and policy networks to guide the exploration of the state-action space. This approach has led to superhuman performance in various games such as Go and StarCraft, and even discovered [faster matrix multiplication algorithms](https://www.nature.com/articles/s41586-022-05172-4).

In this guide, we demonstrate how to apply the mlpack implementation of AlphaZero on the benchmark task, CartPole.

## Environment
AlphaZero can be applied to any environment with a discrete action space. However, as AlphaZero simulates state-aciton transitions, the number of steps an agent can perform must be unbounded. In consequence, the environment’s `maxStep` must be set to `0` as this nutraliases the upper bound on the maximum number of transitions in mlpack environments.

AlphaZero can be applied to any environment with a **discrete action space**. However, since AlphaZero relies on simulating state-action transitions, the number of steps an agent can perform must be unbounded. To achieve this in `mlpack` environments, set the environment's `maxStep` to `0`, which removes the upper bound on the number of transitions.


```c++
int main() {
  /* Setup the environment */
  mlpack::CartPole environment(
     0 // maxSteps = 0 removes the limit on the number of transitions
  );
```

## Networks

AlphaZero explores the state-action space using two neural networks the **value network** and the **policy network**.

- The **value network** predicts the expected return from a given state.
- The **policy network** predicts which action is most likely to lead to a high return.

The returns are scaled between `-1` and `1`, so the value network's output must **always** fall within this range. Any activation function ensuring this property, such as `TanH`, can be used. Furthermore, the policy network must end with a `LogSoftMax` layer, which is more computationally stable and efficient than a softmax layer. Both networks should use `EmptyLoss` as the loss function.

```c++
  /* Set up the value network */
  mlpack::FFN<mlpack::EmptyLoss, mlpack::GaussianInitialization> network_v(mlpack::EmptyLoss(),
            mlpack::GaussianInitialization(0, 0.1));
  network_v.Add<mlpack::Linear>(64);
  network_v.Add<mlpack::ReLU>();
  network_v.Add<mlpack::Linear>(64);
  network_v.Add<mlpack::ReLU>();
  network_v.Add<mlpack::Linear>(1);
  network_v.Add<mlpack::TanH>(); // Ensures output is between -1 and 1

  /* Set up the policy network */
  mlpack::FFN<mlpack::EmptyLoss, mlpack::GaussianInitialization> network_p(mlpack::EmptyLoss(),
            mlpack::GaussianInitialization(0, 0.1));
  network_p.Add<mlpack::Linear>(64);
  network_p.Add<mlpack::ReLU>();
  network_p.Add<mlpack::Linear>(64);
  network_p.Add<mlpack::ReLU>();
  network_p.Add<mlpack::Linear>(2); // Action space dimension for CartPole is 2
  network_p.Add<mlpack::LogSoftMax>(); // Ensures stability and efficiency in computations
```

## Replay Method
Properly defining the replay method is key to use mlpack AlphaZero implementation. First, AlphaZero can only be coupled with `RandomReplay`, and the second template argument must be `true` to indicate that the replay method does not store actions but probability vectors. Second, the argument `backpropagate` must be set to true in order to replace each move reward with the returned obtained at the end of the episode.

Correctly configuring the replay method is essential for AlphaZero in `mlpack`.

1. AlphaZero is only compatible with `RandomReplay`.
2. The second template argument must be set to `true` to indicate that action probability vectors, rather than the actions themselves, are stored.
3. The backpropagate argument must be `true`, as it replaces each move’s reward with the final return of the episode.
```c++
mlpack::RandomReplay<mlpack::CartPole, 
  true // saveProba, default is false
  > 
  replayMethod(
    128, // batch size
    10000, // capacity
    1, // nSteps (should remain 1 for AlphaZero)
    true // backpropagate to replace move rewards with final returns
  );
```

## Configuration

AlphaZero has a configuration class with several important parameters that need to be set:
```c++
  /* Set up the AlphaZero configuration */
  mlpack::AlphaZeroConfig config;
  config.StepSize() = 0.01; // Learning rate for the value and policy networks
  config.ExplorationSteps() = 5; // Episodes before the first network update
  config.StepLimit() = 200; // Maximum number of episodes for training
  config.ExplorationUCB() = 1.0; // Exploration constant (default is √2)
  config.SelfplaySteps() = 200; // Number of exploration steps
  config.UpdateInterval() = 1; // Frequency of network updates
  config.VMin() = 1.0; // Minimum possible return of the environment
  config.VMax() = 500.0; // Maximum possible return of the environment
```
Because AlphaZero scales returns between `VMin` and `VMax`, you must specify the minimum and maximum possible returns for your environment. If the returns are unbounded (as in CartPole), you should define a satisfactory threshold.

## Training AlphaZero
After setting up the environment, networks, replay method, and configuration, we can assemble everything easily:
```c++
  AlphaZero<mlpack::CartPole, decltype(network_v), decltype(network_p), ens::AdamUpdate>
  agent(config, network_v, network_p, replayMethod, environment);
```

Now, we define the training loop to allow AlphaZero to learn:

```c++
  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > config.StepLimit())
    {
        std::cout << "Cart Pole with AlphaLike failed." << std::endl;
        converged = false;
        break;
    }
    std::cout << "Average return: " << averageReturn.mean() 
                << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 35) // A running average return of 35 is enough to show convergence
      break;
  }
  if (converged)
    std::cout << "Hooray! AlphaZero agent successfully trained" << std::endl;
```
AlphaZero has solved problems that scientists had thought impossible. Good look with yours!

## Full Example
Below is the complete code for the tutorial:
```c++
#include <mlpack.hpp>

int main()
{
  mlpack::RandomSeed(time(NULL));

  /* Setup the environment */
  mlpack::CartPole environment(
    0 // maxSteps = 0 removes the limit on the number of transitions
  )  
  /* Set up the value network */
  mlpack::FFN<mlpack::EmptyLoss, mlpack::GaussianInitialization> network_v(mlpack::EmptyLoss(),
            mlpack::GaussianInitialization(0, 0.1)
  );
  network_v.Add<mlpack::Linear>(64);
  network_v.Add<mlpack::ReLU>();
  network_v.Add<mlpack::Linear>(64);
  network_v.Add<mlpack::ReLU>();
  network_v.Add<mlpack::Linear>(1);
  network_v.Add<mlpack::TanH>(); // Ensures output is between -1 and 1

  /* Set up the policy network */
  mlpack::FFN<mlpack::EmptyLoss, mlpack::GaussianInitialization> network_p(mlpack::EmptyLoss(),
            mlpack::GaussianInitialization(0, 0.1)
  );
  network_p.Add<mlpack::Linear>(64);
  network_p.Add<mlpack::ReLU>();
  network_p.Add<mlpack::Linear>(64);
  network_p.Add<mlpack::ReLU>();
  network_p.Add<mlpack::Linear>(2); // Action space dimension for CartPole is 2
  network_p.Add<mlpack::LogSoftMax>(); // Ensures stability and efficiency in computations

  // Set up the replay method
    mlpack::RandomReplay<mlpack::CartPole, 
    true // saveProba, default is false
    > replayMethod(
        128, // batch size
        10000, // capacity
        1, // nSteps (should remain 1 for AlphaZero)
        true // backpropagate to replace move rewards with final returns
    );
    
  /* Set up the AlphaZero configuration */
  mlpack::AlphaZeroConfig config;
  config.StepSize() = 0.01; // Learning rate for the value and policy networks
  config.ExplorationSteps() = 5; // Episodes before the first network update
  config.StepLimit() = 200; // Maximum number of episodes for training
  config.ExplorationUCB() = 1.0; // Exploration constant (default is √2)
  config.SelfplaySteps() = 200; // Number of exploration steps
  config.UpdateInterval() = 1; // Frequency of network updates
  config.VMin() = 1.0; // Minimum possible return of the environment
  config.VMax() = 500.0; // Maximum possible return of the environment

  /* Declare AlphaZero */
  AlphaZero<mlpack::CartPole, decltype(network_v), decltype(network_p), ens::AdamUpdate>
  agent(config, network_v, network_p, replayMethod, environment);

/* Defining the training loop */
  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > config.StepLimit())
    {
      std::cout << "Cart Pole with AlphaLike failed." << std::endl;
      converged = false;
      break;
    }
    std::cout << "Average return: " << averageReturn.mean() 
                << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 35) // A running average return of 35 is enough to show convergence
      break;
  }
  if (converged)
    std::cout << "Hooray! AlphaZero agent successfully trained" << std::endl;

  return 0;
}
```
