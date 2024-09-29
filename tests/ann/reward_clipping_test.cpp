/**
 * @file tests/reward_clipping_test.cpp
 * @author Shashank Shekhar
 *
 * Test for the reward clipping wrapper for reinforcement learning environments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/reinforcement_learning/reinforcement_learning.hpp>

#include "../catch.hpp"

using namespace mlpack;
using namespace ens;

// Test checking that reward clipping works with vanilla update.
TEST_CASE("ClippedRewardTest", "[RewardClippingTest]")
{
  Pendulum task;
  RewardClipping<Pendulum> rewardClipping(task, -2.0, +2.0);

  RewardClipping<Pendulum>::State state = rewardClipping.InitialSample();
  RewardClipping<Pendulum>::Action action;
  action.action[0] = Random(-1.0, 1.0);
  double reward = rewardClipping.Sample(state, action);

  REQUIRE(reward <= 2.0);
  REQUIRE(reward >= -2.0);
}

//! Test DQN in Acrobot task.
TEST_CASE("RewardClippedAcrobotWithDQN", "[RewardClippingTest]")
{
  // We will allow three trials, although it would be very uncommon for the test
  // to use more than one.
  bool converged = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    // Set up the network.
    SimpleDQN<> model(64, 32, 3);

    // Set up the policy and replay method.
    GreedyPolicy<RewardClipping<Acrobot>> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<RewardClipping<Acrobot>> replayMethod(20, 10000);

    // Set up Acrobot task and reward clipping wrapper.
    Acrobot task;
    RewardClipping<Acrobot> rewardClipping(task, -2.0, +2.0);

    // Set up update rule.
    AdamUpdate update;

    TrainingConfig config;
    config.StepSize() = 0.01;
    config.Discount() = 0.99;
    config.TargetNetworkSyncInterval() = 100;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = false;
    config.StepLimit() = 400;

    // Set up DQN agent.
    QLearning<decltype(rewardClipping), decltype(model), AdamUpdate,
              decltype(policy)>
        agent(config, model, policy, replayMethod, std::move(update),
        std::move(rewardClipping));

    arma::running_stat<double> averageReturn;
    size_t episodes = 0;
    converged = true;
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
       * I am using a thresold of -380 to check convegence.
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
      break;
  }

  REQUIRE(converged);
}
