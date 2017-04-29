/**
 * @file rl_environment_test.hpp
 * @author Shangtong Zhang
 *
 * Basic test for the reinforcement learning task environment
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(RLEnvironmentTest)

BOOST_AUTO_TEST_CASE(MountainCarTest)
{
  const auto task = MountainCar();
  auto state = task.InitialSample();
  auto action = MountainCar::Action::backward;
  auto reward = task.Sample(state, action);
  BOOST_REQUIRE(!task.IsTerminal(state));
  BOOST_REQUIRE_EQUAL(3, MountainCar::Action::size);
}

BOOST_AUTO_TEST_CASE(CartPoleTest)
{
  const auto task = CartPole();
  auto state = task.InitialSample();
  auto action = CartPole::Action::backward;
  auto reward = task.Sample(state, action);
  BOOST_REQUIRE(!task.IsTerminal(state));
  BOOST_REQUIRE_EQUAL(2, CartPole::Action::size);
}

BOOST_AUTO_TEST_SUITE_END()
