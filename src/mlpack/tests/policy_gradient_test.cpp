/**
 * @file tests/policy_gradient_test.cpp
 * @author Tarek Elsayed
 *
 * Test for reinforcement learning policy gradient algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>
#include <mlpack/methods/reinforcement_learning.hpp>

#include "catch.hpp"
#include "test_reinforcement_learning_agent.hpp"

using namespace mlpack;
using namespace ens;

//! Test DDPG on Pendulum task.
TEST_CASE("PendulumWithDDPG", "[PolicyGradientTest]")
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 8; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;
    // Set up the replay method.
    RandomReplay<Pendulum> replayMethod(32, 10000);

    TrainingConfig config;
    config.StepSize() = 0.001;
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
    double sigma = 0.01;

    // Create an instance of the OUNoise class.
    OUNoise ouNoise(size, mu, theta, sigma);

    // Set up Deep Deterministic Policy Gradient agent.
    DDPG<Pendulum, decltype(qNetwork), decltype(policyNetwork),
        OUNoise, AdamUpdate>
        agent(config, qNetwork, policyNetwork, ouNoise, replayMethod);

    converged = testAgent<decltype(agent)>(agent, -900, 500, 10);
    if (converged)
      break;
  }
  REQUIRE(converged);
}

//! Test DDPG on Pendulum task with Gaussian noise.
TEST_CASE("PendulumWithGaussianDDPG", "[PolicyGradientTest]")
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 8; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;
    // Set up the replay method.
    RandomReplay<Pendulum> replayMethod(32, 10000);

    TrainingConfig config;
    config.StepSize() = 0.001;
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

    // Set up the GaussianNoise parameters.
    int size = 1;
    double mu = 0.0;
    double sigma = 0.01;

    // Create an instance of the GaussianNoise class.
    GaussianNoise gaussianNoise(size, mu, sigma);

    // Set up Deep Deterministic Policy Gradient agent.
    DDPG<Pendulum, decltype(qNetwork), decltype(policyNetwork),
        GaussianNoise, AdamUpdate>
        agent(config, qNetwork, policyNetwork, gaussianNoise, replayMethod);

    converged = testAgent<decltype(agent)>(agent, -900, 500, 10);
    if (converged)
      break;
  }
  REQUIRE(converged);
}

//! A test to ensure DDPG works with multiple actions in action space.
TEST_CASE("DDPGForMultipleActions", "[PolicyGradientTest]")
{
  FFN<EmptyLoss, GaussianInitialization>
      policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear(128));
  policyNetwork.Add(new ReLU());
  policyNetwork.Add(new Linear(4));
  policyNetwork.Add(new TanH());

  FFN<EmptyLoss, GaussianInitialization>
      qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear(128));
  qNetwork.Add(new ReLU());
  qNetwork.Add(new Linear(1));

  // Set up the replay method.
  RandomReplay<ContinuousActionEnv<3, 4>> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;

  // Set up the OUNoise parameters.
  int size = 4;
  double mu = 0.0;
  double theta = 1.0;
  double sigma = 0.01;

  // Create an instance of the OUNoise class.
  OUNoise ouNoise(size, mu, theta, sigma);

  // Set up the DDPG agent.
  DDPG<ContinuousActionEnv<3, 4>, decltype(qNetwork), decltype(policyNetwork),
      OUNoise, AdamUpdate>
      agent(config, qNetwork, policyNetwork, ouNoise, replayMethod);

  agent.State().Data() = arma::randu<arma::colvec>
      (ContinuousActionEnv<3, 4>::State::dimension, 1);
  agent.SelectAction();

  // Test to check if the action dimension given by the agent is correct.
  REQUIRE(agent.Action().action.size() == 4);

  replayMethod.Store(agent.State(), agent.Action(), 1, agent.State(), 1, 0.99);
  agent.TotalSteps()++;
  agent.Update();
  // If the agent is able to reach this point of the test, it is assured
  // that the agent can handle multiple actions in continuous space.
}

//! Test Ornstein-Uhlenbeck noise class.
TEST_CASE("OUNoiseTest", "[PolicyGradientTest]")
{
  // Set up the OUNoise parameters.
  int size = 3;
  double mu = 0.0;
  double theta = 0.15;
  double sigma = 0.02;

  // Create an instance of the OUNoise class.
  OUNoise ouNoise(size, mu, theta, sigma);

  // Test the reset function.
  ouNoise.reset();
  arma::colvec state = ouNoise.sample();
  REQUIRE((int) state.n_elem == size);

  // Verify that the sample is not equal to the reset state.
  arma::colvec sample = ouNoise.sample();
  bool isNotEqual = arma::any(sample != state);
  REQUIRE(isNotEqual);
}

//! Test Gaussian noise class.
TEST_CASE("GaussianNoiseTest", "[PolicyGradientTest]")
{
  // Set up the GaussianNoise parameters.
  int size = 5;
  double mu = 0.0;
  double sigma = 0.00001;

  // Create an instance of the GaussianNoise class.
  GaussianNoise gaussianNoise(size, mu, sigma);

  // Test the sample function.
  arma::colvec noise = gaussianNoise.sample();
  REQUIRE((int) noise.n_elem == size);

  // Verify that the noise vector has values drawn from a
  // Gaussian distribution with the specified mean and standard deviation.
  double mean = arma::mean(noise);
  double stdDev = arma::stddev(noise);

  double meanErr = mean - mu;
  double stdDevErr = stdDev - sigma;
  REQUIRE(meanErr <= 1e-4);
  REQUIRE(stdDevErr <= 1e-4);
}

//! Test TD3 on Pendulum task.
TEST_CASE("PendulumWithTD3", "[PolicyGradientTest]")
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 8; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;
    // Set up the replay method.
    RandomReplay<Pendulum> replayMethod(32, 10000);

    TrainingConfig config;
    config.StepSize() = 0.001;
    config.TargetNetworkSyncInterval() = 2;
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

    // Set up Twin Delayed Deep Deterministic policy gradient agent.
    TD3<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
        agent(config, qNetwork, policyNetwork, replayMethod);

    converged = testAgent<decltype(agent)>(agent, -900, 500, 10);
    if (converged)
      break;
  }
  REQUIRE(converged);
}

//! A test to ensure TD3 works with multiple actions in action space.
TEST_CASE("TD3ForMultipleActions", "[PolicyGradientTest]")
{
  FFN<EmptyLoss, GaussianInitialization>
      policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear(128));
  policyNetwork.Add(new ReLU());
  policyNetwork.Add(new Linear(4));
  policyNetwork.Add(new TanH());

  FFN<EmptyLoss, GaussianInitialization>
      qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear(128));
  qNetwork.Add(new ReLU());
  qNetwork.Add(new Linear(1));

  // Set up the replay method.
  RandomReplay<ContinuousActionEnv<3, 4>> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.TargetNetworkSyncInterval() = 2;
  config.UpdateInterval() = 3;

  // Set up the TD3 agent.
  TD3<ContinuousActionEnv<3, 4>, decltype(qNetwork), decltype(policyNetwork),
      AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  agent.State().Data() = arma::randu<arma::colvec>
      (ContinuousActionEnv<3, 4>::State::dimension, 1);
  agent.SelectAction();

  // Test to check if the action dimension given by the agent is correct.
  REQUIRE(agent.Action().action.size() == 4);

  replayMethod.Store(agent.State(), agent.Action(), 1, agent.State(), 1, 0.99);
  agent.TotalSteps()++;
  agent.Update();
  // If the agent is able to reach this point of the test, it is assured
  // that the agent can handle multiple actions in continuous space.
}

//! Test SAC on Pendulum task.
TEST_CASE("PendulumWithSAC", "[PolicyGradientTest]")
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 8; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;
    // Set up the replay method.
    RandomReplay<Pendulum> replayMethod(32, 10000);

    TrainingConfig config;
    config.StepSize() = 0.001;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;

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

    converged = testAgent<decltype(agent)>(agent, -900, 500, 10);
    if (converged)
      break;
  }
  REQUIRE(converged);
}

//! A test to ensure SAC works with multiple actions in action space.
TEST_CASE("SACForMultipleActions", "[PolicyGradientTest]")
{
  FFN<EmptyLoss, GaussianInitialization>
      policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear(128));
  policyNetwork.Add(new ReLU());
  policyNetwork.Add(new Linear(4));
  policyNetwork.Add(new TanH());

  FFN<EmptyLoss, GaussianInitialization>
      qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear(128));
  qNetwork.Add(new ReLU());
  qNetwork.Add(new Linear(1));

  // Set up the replay method.
  RandomReplay<ContinuousActionEnv<3, 4>> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;

  // Set up Soft actor-critic agent.
  SAC<ContinuousActionEnv<3, 4>, decltype(qNetwork), decltype(policyNetwork),
      AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  agent.State().Data() = arma::randu<arma::colvec>
      (ContinuousActionEnv<3, 4>::State::dimension, 1);
  agent.SelectAction();

  // Test to check if the action dimension given by the agent is correct.
  REQUIRE(agent.Action().action.size() == 4);

  replayMethod.Store(agent.State(), agent.Action(), 1, agent.State(), 1, 0.99);
  agent.TotalSteps()++;
  agent.Update();
  // If the agent is able to reach till this point of the test, it is assured
  // that the agent can handle multiple actions in continuous space.
}
