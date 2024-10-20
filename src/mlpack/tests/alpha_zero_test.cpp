/**
 * @file tests/alpha_zero_test.cpp
 * @author Antoine Dubois
 *
 * Test for AlphaZero implementation
 *
 * s free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>
#include <mlpack/methods/reinforcement_learning.hpp>

#include "catch.hpp"
#include "test_reinforcement_learning_agent.hpp"

using namespace mlpack;
using namespace ens;

//! Test DQN in Cart Pole task.
TEST_CASE("CartPoleWithAlphaZero", "[AlphaZeroTest]")
{
  /* Setup the environment */
  CartPole environment(
    0 // maxSteps = 0 removes the limit on the number of transitions
  );
  /* Set up the value network */
  FFN<EmptyLoss, GaussianInitialization> network_v(EmptyLoss(),
            GaussianInitialization(0, 0.1)
  );
  network_v.Add<Linear>(64);
  network_v.Add<ReLU>();
  network_v.Add<Linear>(64);
  network_v.Add<ReLU>();
  network_v.Add<Linear>(1);
  network_v.Add<TanH>(); // Ensures output is between -1 and 1

  /* Set up the policy network */
  FFN<EmptyLoss, GaussianInitialization> network_p(EmptyLoss(),
            GaussianInitialization(0, 0.1)
  );
  network_p.Add<Linear>(64);
  network_p.Add<ReLU>();
  network_p.Add<Linear>(64);
  network_p.Add<ReLU>();
  network_p.Add<Linear>(2); // Action space dimension for CartPole is 2
  network_p.Add<LogSoftMax>(); // Ensures stability and efficiency in computations

  // Set up the replay method
  RandomReplay<CartPole, 
    true // saveProba, default is false
  > replayMethod(
      128, // batch size
      10000, // capacity
      1, // nSteps (should remain 1 for AlphaZero)
      true // backpropagate to replace move rewards with final returns
  );
    
  /* Set up the AlphaZero configuration */
  AlphaZeroConfig config;
  config.StepSize() = 0.01; // Learning rate for the value and policy networks
  config.ExplorationSteps() = 5; // Episodes before the first network update
  config.StepLimit() = 200; // Maximum number of episodes for training
  config.ExplorationUCB() = 1.0; // Exploration constant (default is √2)
  config.SelfplaySteps() = 200; // Number of exploration steps
  config.UpdateInterval() = 1; // Frequency of network updates
  config.VMin() = 1.0; // Minimum possible return of the environment
  config.VMax() = 500.0; // Maximum possible return of the environment

  /* Declare AlphaZero */
  AlphaZero<CartPole, decltype(network_v), decltype(network_p), ens::AdamUpdate>
  agent(config, network_v, network_p, replayMethod, environment);


  bool converged = testAgent<decltype(agent)>(agent, 35, 200);

  // To check if the action returned by the agent is not nan and is finite.
  REQUIRE(std::isfinite(double(agent.Action().action)));
  REQUIRE(converged);
}

