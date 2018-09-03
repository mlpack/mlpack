/**
 * @file reward_clipping_test.hpp
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

#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/continuous_mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobat.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>
#include <mlpack/methods/reinforcement_learning/environment/reward_clipping.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(RewardClippingTest);

// Test checking that reward clipping works with vanilla update.
BOOST_AUTO_TEST_CASE(ClippedRewardTest)
{
  Pendulum task;
  RewardClipping<Pendulum> rewardClipping(task, -2.0, +2.0);
  
  RewardClipping<Pendulum>::State state = rewardClipping.InitialSample();
  RewardClipping<Pendulum>::Action action;
  action.action[0] = math::Random(-1.0, 1.0);
  double reward = rewardClipping.Sample(state, action);

  BOOST_REQUIRE(reward <= 2.0);
  BOOST_REQUIRE(reward >= -2.0);
}

BOOST_AUTO_TEST_SUITE_END();