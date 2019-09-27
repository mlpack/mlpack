/**
 * @file q_learning_test.hpp
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
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(QLearningTest);

//! Test DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDQN)
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
      GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);

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

  // Set up DQN agent.
  QLearning<CartPole, decltype(model), AdamUpdate, decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy),
      std::move(replayMethod));

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
      Log::Debug << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }

    /**
     * Reaching running average return 35 is enough to show it works.
     * For the speed of the test case, I didn't set high criterion.
     */
    Log::Debug << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 35)
    {
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }
  }
  BOOST_REQUIRE(converged);
}

//! Test DQN in Cart Pole task with Prioritized Replay.
BOOST_AUTO_TEST_CASE(CartPoleWithDQNPrioritizedReplay)
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
      GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
  PrioritizedReplay<CartPole> replayMethod(10, 10000, 0.6);

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.9;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 100;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;

  // Set up DQN agent.
  QLearning<CartPole, decltype(model), AdamUpdate, decltype(policy),
      decltype(replayMethod)>
      agent(std::move(config), std::move(model), std::move(policy),
          std::move(replayMethod));

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
      Log::Debug << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }

    /**
     * Reaching running average return 35 is enough to show it works.
     * For the speed of the test case, I didn't set high criterion.
     */
    Log::Debug << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 35)
    {
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }
  }

  BOOST_REQUIRE(converged);
}

//! Test Double DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDoubleDQN)
{
  // It isn't guaranteed that the network will converge in the specified number
  // of iterations using random weights. If this works 1 of 4 times, I'm fine
  // with that.
  size_t episodes = 0;
  bool converged = false;
  for (size_t trial = 0; trial < 4; ++trial)
  {
    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
        GaussianInitialization(0, 0.001));
    model.Add<Linear<>>(4, 20);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(20, 20);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(20, 2);

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

    // Set up the DQN agent.
    QLearning<CartPole, decltype(model), RMSPropUpdate, decltype(policy)>
        agent(std::move(config), std::move(model), std::move(policy),
        std::move(replayMethod));

    arma::running_stat<double> averageReturn;

    for (episodes = 0; episodes <= 1000; ++episodes)
    {
      double episodeReturn = agent.Episode();
      averageReturn(episodeReturn);

      /**
       * Reaching running average return 35 is enough to show it works.
       * For the speed of the test case, I didn't set high criterion.
       */
      Log::Debug << "Average return: " << averageReturn.mean()
          << " Episode return: " << episodeReturn << std::endl;
      if (averageReturn.mean() > 40)
      {
        agent.Deterministic() = true;
        arma::running_stat<double> testReturn;
        for (size_t i = 0; i < 10; ++i)
          testReturn(agent.Episode());
        Log::Debug << "Average return in deterministic test: "
            << testReturn.mean() << std::endl;
        break;
      }
    }

    if (episodes < 1000)
    {
      converged = true;
      break;
    }
  }

  BOOST_REQUIRE(converged);
}

//! Test DQN in Acrobot task.
BOOST_AUTO_TEST_CASE(AcrobotWithDQN)
{
  // We will allow three trials, although it would be very uncommon for the test
  // to use more than one.
  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
        GaussianInitialization(0, 0.001));
    model.Add<Linear<>>(4, 64);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(64, 32);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(32, 3);

    // Set up the policy and replay method.
    GreedyPolicy<Acrobot> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<Acrobot> replayMethod(20, 10000);

    TrainingConfig config;
    config.StepSize() = 0.01;
    config.Discount() = 0.99;
    config.TargetNetworkSyncInterval() = 100;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = false;
    config.StepLimit() = 400;

    // Set up DQN agent.
    QLearning<Acrobot, decltype(model), AdamUpdate, decltype(policy)>
        agent(std::move(config), std::move(model), std::move(policy),
        std::move(replayMethod));

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
        Log::Debug << "Acrobot with DQN failed." << std::endl;
        converged = false;
        break;
      }

      /**
       * I am using a threshold of -380 to check convergence.
       */
      Log::Debug << "Average return: " << averageReturn.mean()
          << " Episode return: " << episodeReturn << std::endl;
      if (averageReturn.mean() > -380.00)
      {
        agent.Deterministic() = true;
        arma::running_stat<double> testReturn;
        for (size_t i = 0; i < 20; ++i)
          testReturn(agent.Episode());

        Log::Debug << "Average return in deterministic test: "
            << testReturn.mean() << std::endl;
        break;
      }
    }

    if (converged)
    {
      success = true;
      break;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

//! Test DQN in Mountain Car task.
BOOST_AUTO_TEST_CASE(MountainCarWithDQN)
{
  // We will allow three trials total.
  bool success = false;
  for (size_t trial = 0; trial < 3; trial++)
  {
    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
        GaussianInitialization(0, 0.001));
    model.Add<Linear<>>(2, 64);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(64, 32);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(32, 3);

    // Set up the policy and replay method.
    GreedyPolicy<MountainCar> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<MountainCar> replayMethod(20, 10000);

    TrainingConfig config;
    config.StepSize() = 0.0001;
    config.Discount() = 0.9;
    config.TargetNetworkSyncInterval() = 100;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = false;
    config.StepLimit() = 400;

    // Set up DQN agent.
    QLearning<MountainCar, decltype(model), AdamUpdate, decltype(policy)>
        agent(std::move(config), std::move(model), std::move(policy),
        std::move(replayMethod));

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
        Log::Debug << "Mountain Car with DQN failed." << std::endl;
        converged = false;
        break;
      }

      /**
       * Set a threshold of -370 to check convergence.
       */
      Log::Debug << "Average return: " << averageReturn.mean()
          << " Episode return: " << episodeReturn << std::endl;
      if (averageReturn.mean() > -370)
      {
        agent.Deterministic() = true;
        arma::running_stat<double> testReturn;
        for (size_t i = 0; i < 10; ++i)
          testReturn(agent.Episode());

        Log::Debug << "Average return in deterministic test: "
            << testReturn.mean() << std::endl;
        break;
      }
    }

    if (converged)
    {
      success = true;
      break;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

//! Test DQN in DoublePoleCart task.
BOOST_AUTO_TEST_CASE(DoublePoleCartWithDQN)
{
  // We will allow four trials total.
  bool success = false;
  for (size_t trial = 0; trial < 4; trial++)
  {
    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
        GaussianInitialization(0, 0.001));
    model.Add<Linear<>>(6, 256);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(256, 3);

    // Set up the policy and replay method.
    GreedyPolicy<DoublePoleCart> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<DoublePoleCart> replayMethod(20, 10000);

    TrainingConfig config;
    config.StepSize() = 0.01;
    config.Discount() = 0.9;
    config.TargetNetworkSyncInterval() = 100;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = false;
    config.StepLimit() = 600;

    // Set up DQN agent.
    QLearning<DoublePoleCart, decltype(model), AdamUpdate, decltype(policy)>
        agent(std::move(config), std::move(model), std::move(policy),
        std::move(replayMethod));

    size_t episodes = 0;
    bool converged = true;
    size_t episodeSuccesses = 0;
    while (true)
    {
      double episodeReturn = agent.Episode();
      episodes += 1;

      if (episodeReturn >= 280)
        episodeSuccesses++;

      if (episodes > 2000)
      {
        Log::Debug << "Cart Pole with DQN failed." << std::endl;
        converged = false;
        break;
      }

      // If the model can solve the environment in two trials this is fine for
      // a simple test.
      if (episodeSuccesses >= 2)
      {
        Log::Debug << "QLearning has succeeded in the multiple pole cart" <<
            " environment." << std::endl;
        converged = true;
        break;
      }
    }

    if (converged)
    {
      success = true;
      break;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
}

BOOST_AUTO_TEST_SUITE_END();
