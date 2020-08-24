/**
 * @file tests/q_learning_test.cpp
 * @author Shangtong Zhang
 * @author Rohan Raj
 *
 * Test for Q-Learning implementation
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/sac.hpp>
#include <mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp>
#include <mlpack/methods/reinforcement_learning/q_networks/dueling_dqn.hpp>
#include <mlpack/methods/reinforcement_learning/q_networks/categorical_dqn.hpp>
#include <mlpack/methods/reinforcement_learning/environment/env_type.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

#include <ensmallen.hpp>
#include <numeric>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(QLearningTest);

template<typename AgentType>
bool testAgent(AgentType& agent,
               const double rewardThreshold,
               const size_t noOfEpisodes,
               const size_t consecutiveEpisodesTest = 50)
{
  bool converged = false;
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

    Log::Debug << "Average return in last " << returnList.size()
        << " consecutive episodes: " << averageReturn
        << " Episode return: " << episodeReturn << std::endl;

    // For the speed of the test case, a high criterion should not be set
    // for the rewardThreshold.
    if (averageReturn > rewardThreshold &&
        returnList.size() >= consecutiveEpisodesTest)
    {
      converged = true;
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }

    if (episodes > noOfEpisodes)
    {
      Log::Debug << "Agent failed." << std::endl;
      break;
    }
  }

  return converged;
}

//! Test DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDQN)
{
  // Set up the network.
  SimpleDQN<> network(4, 128, 128, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<CartPole> replayMethod(10, 10000);

  // Setting all training hyperparameters.
  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.9;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
      agent(config, network, policy, replayMethod);

  bool converged = testAgent<decltype(agent)>(agent, 40, 1000);

  // To check if the action returned by the agent is not nan and is finite.
  BOOST_REQUIRE(std::isfinite(double(agent.Action().action)));
  BOOST_REQUIRE(converged);
}

//! Test DQN in Cart Pole task with Prioritized Replay.
BOOST_AUTO_TEST_CASE(CartPoleWithDQNPrioritizedReplay)
{
  // Set up the network.
  SimpleDQN<> network(4, 128, 128, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
  PrioritizedReplay<CartPole> replayMethod(10, 10000, 0.6);

  TrainingConfig config;
  config.ExplorationSteps() = 100;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy),
      decltype(replayMethod)>
      agent(config, network, policy, replayMethod);

  bool converged = testAgent<decltype(agent)>(agent, 45, 1000);
  BOOST_REQUIRE(converged);
}

//! Test Double DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDoubleDQN)
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights. If this works 1 of 4 times, I'm fine
  // with that.
  bool converged = false;
  for (size_t trial = 0; trial < 4; ++trial)
  {
    // Set up the network.
    SimpleDQN<> network(4, 20, 20, 2);

    // Set up the policy and replay method.
    GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<CartPole> replayMethod(10, 10000);

    TrainingConfig config;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = true;
    config.StepLimit() = 200;

    // Set up the DQN agent.
    QLearning<CartPole, decltype(network), RMSPropUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    converged = testAgent<decltype(agent)>(agent, 45, 1000);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test DQN in Acrobot task.
BOOST_AUTO_TEST_CASE(AcrobotWithDQN)
{
  // We will allow three trials, although it would be very uncommon for the test
  // to use more than one.
  bool converged = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    // Set up the network.
    SimpleDQN<> network(4, 64, 32, 3);

    // Set up the policy and replay method.
    GreedyPolicy<Acrobot> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<Acrobot> replayMethod(20, 10000);

    TrainingConfig config;
    config.ExplorationSteps() = 100;
    config.StepLimit() = 400;

    // Set up DQN agent.
    QLearning<Acrobot, decltype(network), AdamUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    converged = testAgent<decltype(agent)>(agent, -380, 1000);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test DQN in Mountain Car task.
BOOST_AUTO_TEST_CASE(MountainCarWithDQN)
{
  // We will allow five trials total.
  bool converged = false;
  for (size_t trial = 0; trial < 5; trial++)
  {
    // Set up the network.
    SimpleDQN<> network(2, 64, 32, 3);

    // Set up the policy and replay method.
    GreedyPolicy<MountainCar> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<MountainCar> replayMethod(20, 10000);

    TrainingConfig config;
    config.StepSize() = 0.0001;
    config.ExplorationSteps() = 100;
    config.StepLimit() = 400;

    // Set up DQN agent.
    QLearning<MountainCar, decltype(network), AdamUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    converged = testAgent<decltype(agent)>(agent, -380, 1000);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test DQN in DoublePoleCart task.
BOOST_AUTO_TEST_CASE(DoublePoleCartWithDQN)
{
  bool converged = false;
  // We will allow four trials total.
  for (size_t trial = 0; trial < 4; trial++)
  {
    // Set up the module. Note that we use a custom network here.
    FFN<MeanSquaredError<>, GaussianInitialization> module(
        MeanSquaredError<>(), GaussianInitialization(0, 0.001));
    module.Add<Linear<>>(6, 256);
    module.Add<ReLULayer<>>();
    module.Add<Linear<>>(256, 3);

    // Adding the module to the SimpleDQN network containing required functions.
    SimpleDQN<> network(module);

    // Set up the policy and replay method.
    GreedyPolicy<DoublePoleCart> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<DoublePoleCart> replayMethod(20, 10000);

    TrainingConfig config;
    config.ExplorationSteps() = 100;
    config.StepLimit() = 600;

    // Set up DQN agent.
    QLearning<DoublePoleCart, decltype(network), AdamUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    size_t episodes = 0;
    size_t episodeSuccesses = 0;
    while (true)
    {
      double episodeReturn = agent.Episode();
      episodes += 1;

      if (episodeReturn >= 280)
        episodeSuccesses++;

      if (episodes > 2000)
      {
        Log::Debug << "Agent failed." << std::endl;
        break;
      }

      // If the network can solve the environment in two trials this is fine for
      // a simple test.
      Log::Debug << " Episode return: " << episodeReturn << std::endl;
      if (episodeSuccesses >= 2)
      {
        converged = true;
        Log::Debug << "QLearning has succeeded in the multiple pole cart" <<
            " environment." << std::endl;
        break;
      }
    }
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test Dueling DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDuelingDQN)
{
  // Set up the network.
  DuelingDQN<> network(4, 128, 64, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<CartPole> replayMethod(32, 2000);

  TrainingConfig config;
  config.ExplorationSteps() = 50;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
      agent(config, network, policy, replayMethod);

  bool converged = testAgent<decltype(agent)>(agent, 45, 2000);
  BOOST_REQUIRE(converged);
}

//! Test Dueling DQN in Cart Pole task with Prioritized Replay.
BOOST_AUTO_TEST_CASE(CartPoleWithDuelingDQNPrioritizedReplay)
{
  // Set up the network.
  DuelingDQN<> network(4, 128, 64, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
  PrioritizedReplay<CartPole> replayMethod(32, 2000, 0.6);

  TrainingConfig config;
  config.ExplorationSteps() = 50;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy),
      decltype(replayMethod)>
      agent(config, network, policy, replayMethod);

  bool converged = testAgent<decltype(agent)>(agent, 50, 2000);
  BOOST_REQUIRE(converged);
}

//! Test Noisy DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithNoisyDQN)
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;

    // Set up the policy and replay method.
    GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<CartPole> replayMethod(32, 2000);

    TrainingConfig config;
    config.StepLimit() = 200;
    config.NoisyQLearning() = true;

    // Set up the network with a flag to enable noisy layers.
    SimpleDQN<> network(4, 64, 32, 2, config.NoisyQLearning());

    // Set up DQN agent.
    QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    converged = testAgent<decltype(agent)>(agent, 45, 500, 30);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test Dueling-Double-Noisy DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDuelingDoubleNoisyDQN)
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;

    // Set up the policy and replay method.
    GreedyPolicy<CartPole> policy(1.0, 2000, 0.1, 0.99);
    RandomReplay<CartPole> replayMethod(32, 4000);

    TrainingConfig config;
    config.DoubleQLearning() = true;
    config.StepLimit() = 200;
    config.NoisyQLearning() = true;

    // Set up the network with a flag to enable noisy layers.
    DuelingDQN<> network(4, 64, 64, 2, config.NoisyQLearning());

    // Set up DQN agent.
    QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    converged = testAgent<decltype(agent)>(agent, 45, 500, 30);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test N-step DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithNStepDQN)
{
  // Set up the network.
  SimpleDQN<> network(4, 128, 128, 2);

  // Set up the policy.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  /**
   * For N-step learning, we need to specify n as the last parameter in
   * the replay method. Here we use n = 3.
   */
  RandomReplay<CartPole> replayMethod(10, 10000, 3);

  // Setting all training hyperparameters.
  TrainingConfig config;
  config.ExplorationSteps() = 50;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
      agent(config, network, policy, replayMethod);

  bool converged = testAgent<decltype(agent)>(agent, 50, 1000);
  BOOST_REQUIRE(converged);
}

//! Test N-step Prioritized DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithNStepPrioritizedDQN)
{
  // Set up the network.
  SimpleDQN<> network(4, 128, 128, 2);

  // Set up the policy.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  /**
   * For N-step learning, we need to specify n as the last parameter in
   * the replay method. Here we use n = 3.
   */
  PrioritizedReplay<CartPole> replayMethod(10, 10000, 0.6, 3);

  // Setting all training hyperparameters.
  TrainingConfig config;
  config.ExplorationSteps() = 50;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy),
      decltype(replayMethod)>
      agent(config, network, policy, replayMethod);

  bool converged = testAgent<decltype(agent)>(agent, 50, 1000);
  BOOST_REQUIRE(converged);
}

//! Test Categorical DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithCategoricalDQN)
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations.
  bool converged = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;

    // Set up the policy and replay method.
    GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<CartPole> replayMethod(32, 4000);

    TrainingConfig config;
    config.IsCategorical() = true;
    config.ExplorationSteps() = 32;

    // Set up the module. Note that we use a custom network here.
    FFN<EmptyLoss<>, GaussianInitialization> module(
        EmptyLoss<>(), GaussianInitialization(0, 0.1));
    module.Add<Linear<>>(4, 128);
    module.Add<ReLULayer<>>();
    module.Add<Linear<>>(128, 2 * config.AtomSize());

    // Adding the module to the CategoricalDQN network.
    CategoricalDQN<> network(module, config);

    // Set up DQN agent.
    QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
        agent(config, network, policy, replayMethod);

    converged = testAgent<decltype(agent)>(agent, 40, 1000, 20);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! Test SAC on Pendulum task.
BOOST_AUTO_TEST_CASE(PendulumWithSAC)
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights.
  bool converged = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    Log::Debug << "Trial number: " << trial << std::endl;
    // Set up the replay method.
    RandomReplay<Pendulum> replayMethod(32, 10000);

    TrainingConfig config;
    config.StepSize() = 0.001;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;

    FFN<EmptyLoss<>, GaussianInitialization>
        policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
    policyNetwork.Add(new Linear<>(3, 128));
    policyNetwork.Add(new ReLULayer<>());
    policyNetwork.Add(new Linear<>(128, 1));
    policyNetwork.Add(new TanHLayer<>());

    FFN<EmptyLoss<>, GaussianInitialization>
        qNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
    qNetwork.Add(new Linear<>(3+1, 128));
    qNetwork.Add(new ReLULayer<>());
    qNetwork.Add(new Linear<>(128, 1));

    // Set up Soft actor-critic agent.
    SAC<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
        agent(config, qNetwork, policyNetwork, replayMethod);

    converged = testAgent<decltype(agent)>(agent, -900, 500, 10);
    if (converged)
      break;
  }
  BOOST_REQUIRE(converged);
}

//! A test to ensure SAC works with multiple actions in action space.
BOOST_AUTO_TEST_CASE(SACForMultipleActions)
{
  ContinuousActionEnv::State::dimension = 3;
  ContinuousActionEnv::Action::size = 4;

  FFN<EmptyLoss<>, GaussianInitialization>
      policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension, 128));
  policyNetwork.Add(new ReLULayer<>());
  policyNetwork.Add(new Linear<>(128, ContinuousActionEnv::Action::size));
  policyNetwork.Add(new TanHLayer<>());

  FFN<EmptyLoss<>, GaussianInitialization>
      qNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension +
                            ContinuousActionEnv::Action::size, 128));
  qNetwork.Add(new ReLULayer<>());
  qNetwork.Add(new Linear<>(128, 1));

  // Set up the replay method.
  RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;

  // Set up Soft actor-critic agent.
  SAC<ContinuousActionEnv, decltype(qNetwork), decltype(policyNetwork),
      AdamUpdate>
      agent(config, qNetwork, policyNetwork, replayMethod);

  agent.State().Data() = arma::randu<arma::colvec>
      (ContinuousActionEnv::State::dimension, 1);
  agent.SelectAction();

  // Test to check if the action dimension given by the agent is correct.
  BOOST_REQUIRE_EQUAL(agent.Action().action.size(),
                      ContinuousActionEnv::Action::size);

  replayMethod.Store(agent.State(), agent.Action(), 1, agent.State(), 1, 0.99);
  agent.TotalSteps()++;
  agent.Update();
  // If the agent is able to reach till this point of the test, it is assured
  // that the agent can handle multiple actions in continuous space.
}

BOOST_AUTO_TEST_SUITE_END();
