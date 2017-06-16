/**
 * @file q_learning_test.hpp
 * @author Shangtong Zhang
 *
 * Test for Q-Learning implementation
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/gradient_descent/gradient_descent.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(QLearningTest);

//! Test DQN in Cart Pole task.
BOOST_AUTO_TEST_CASE(CartPoleWithDQN)
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model;
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);

  // Set up the optimizer generator.
  StandardSGD<decltype(model)> opt(model, 0.0001, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
  RandomReplay<CartPole> replayMethod(10, 10000);

  // Set up DQN agent.
  QLearning<CartPole, decltype(model), decltype(opt), decltype(policy)>
      agent(std::move(model), std::move(opt), 0.9, std::move(policy),
          std::move(replayMethod), 100, 100, false, 200);

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
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model;
  model.Add<Linear<>>(4, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 128);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(128, 2);

  // Set up the optimizer.
  StandardSGD<decltype(model)> opt(model, 0.0001, 2);

  // Set up the policy and replay method.
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1);
  RandomReplay<CartPole> replayMethod(10, 10000);

  // Set up the DQN agent.
  QLearning<CartPole, decltype(model), decltype(opt), decltype(policy)>
      agent(std::move(model), std::move(opt), 0.9, std::move(policy),
          std::move(replayMethod), 100, 100, true, 200);

  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;
    if (episodes > 1000) {
      converged = false;
      Log::Debug << "Cart Pole with DQN failed." << std::endl;
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

BOOST_AUTO_TEST_SUITE_END();
