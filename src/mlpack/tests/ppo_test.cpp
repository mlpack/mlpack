/**
 * @file ppo_test.hpp
 * @author Xiaohong Ji and Nishant Kumar
 *
 * Test for PPO implementation
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
#include <mlpack/methods/reinforcement_learning/ppo.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(PPOTEST);

//! Test PPO in CartPole task.
BOOST_AUTO_TEST_CASE(CartPoleWithPPO)
{
  size_t episodes = 0;
  bool converged = false;
  for (size_t trial = 0; trial < 4; ++trial) {
    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> critic(
        MeanSquaredError<>(), GaussianInitialization(0, 0.001));

    critic.Add<Linear<>>(2, 128);
    critic.Add<ReLULayer<>>();
    critic.Add<Linear<>>(128, 1);

    FFN<EmptyLoss<>, GaussianInitialization> actor(
        EmptyLoss<>(), GaussianInitialization(0, 0.001));

    actor.Add<Linear<>>(2, 128);
    actor.Add<ReLULayer<>>();
    actor.Add<Linear<>>(128, 2);

    // Set up the policy and replay method.
    GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<CartPole> replayMethod(10, 10000);

    TrainingConfig config;
    config.StepSize() = 0.001;
    config.Discount() = 0.9;
    config.Epsilon() = 0.2;
    config.StepLimit() = 200;
    config.UpdateInterval() = 32;
    config.ActorUpdateStep() = 10;

    // Set up the PPO agent.
    PPO<CartPole, decltype(actor), decltype(critic), AdamUpdate,
        decltype(policy)>
        agent(std::move(config), std::move(actor), std::move(critic),
        std::move(policy), std::move(replayMethod));

    arma::running_stat<double> averageReturn;

    for (episodes = 0; episodes <= 1000; ++episodes) {
      double episodeReturn = agent.Episode();
      averageReturn(episodeReturn);

      /**
       * I am using a threshold of -136.16 to check convergence.
       */
      Log::Debug << "Average return: " << averageReturn.mean()
                 << " Episode return: " << episodeReturn << std::endl;
      if (averageReturn.mean() > -136.16) {
        agent.Deterministic() = true;
        arma::running_stat<double> testReturn;
        for (size_t i = 0; i < 10; ++i)
          testReturn(agent.Episode());
        Log::Debug << "Average return in deterministic test: "
                   << testReturn.mean() << std::endl;
        break;
      }
    }

    if (episodes < 1000) {
      converged = true;
      break;
    }
  }

  BOOST_REQUIRE(converged);
}

//! Test PPO in Acrobot task.
BOOST_AUTO_TEST_CASE(AcrobotWithPPO)
{
  size_t episodes = 0;
  bool converged = false;
  for (size_t trial = 0; trial < 4; ++trial) {
    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> critic(
      MeanSquaredError<>(), GaussianInitialization(0, 0.001));

    critic.Add<Linear<>>(2, 128);
    critic.Add<ReLULayer<>>();
    critic.Add<Linear<>>(128, 1);

    FFN<EmptyLoss<>, GaussianInitialization> actor(
      EmptyLoss<>(), GaussianInitialization(0, 0.001));

    actor.Add<Linear<>>(2, 128);
    actor.Add<ReLULayer<>>();
    actor.Add<Linear<>>(128, 2);

    // Set up the policy and replay method.
    GreedyPolicy<ContinuousMountainCar> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<ContinuousMountainCar> replayMethod(64, 10000);

    TrainingConfig config;
    config.StepSize() = 0.0001;
    config.Discount() = 0.99;
    config.Epsilon() = 0.2;
    config.StepLimit() = 100000;
    config.UpdateInterval() = 64;
    config.ActorUpdateStep() = 10;

    // Set up the PPO agent.
    PPO<ContinuousMountainCar, decltype(actor), decltype(critic), AdamUpdate,
        decltype(policy)>
        agent(std::move(config), std::move(actor), std::move(critic),
        std::move(policy), std::move(replayMethod));

    arma::running_stat<double> averageReturn;

    for (episodes = 0; episodes <= 1000; ++episodes) {
      double episodeReturn = agent.Episode();
      averageReturn(episodeReturn);

      /**
       * I am using a threshold of 120 to check convergence.
       */
      Log::Debug << "Average return: " << averageReturn.mean()
                 << " Episode return: " << episodeReturn << std::endl;
      if (averageReturn.mean() > 120) {
        agent.Deterministic() = true;
        arma::running_stat<double> testReturn;
        for (size_t i = 0; i < 10; ++i)
          testReturn(agent.Episode());
        Log::Debug << "Average return in deterministic test: "
                   << testReturn.mean() << std::endl;
        break;
      }
    }

    if (episodes < 1000) {
      converged = true;
      break;
    }
  }

  BOOST_REQUIRE(converged);
}

BOOST_AUTO_TEST_SUITE_END();
