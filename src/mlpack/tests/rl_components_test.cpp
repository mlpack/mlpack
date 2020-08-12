/**
 * @file tests/rl_components_test.cpp
 * @author Shangtong Zhang
 * @author Rohan Raj
 *
 * Basic test for the components of reinforcement learning algorithms.
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
#include <mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/environment/continuous_double_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>
#include <mlpack/methods/reinforcement_learning/replay/random_replay.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(RLComponentsTest)

/**
 * Constructs a Pendulum instance and check if the main routine works as it
 * should be working.
 */
BOOST_AUTO_TEST_CASE(SimplePendulumTest)
{
  Pendulum task = Pendulum();
  task.MaxSteps() = 20;

  Pendulum::State state = task.InitialSample();
  Pendulum::Action action;
  action.action[0] = math::Random(-2.0, 2.0);
  double reward, minReward = 0.0;

  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
  {
    reward = task.Sample(state, action, state);
    minReward = std::min(reward, minReward);
  }

  // The reward is always negative. Check if not lower than lowest possible.
  BOOST_REQUIRE(minReward >= -(pow(M_PI, 2) + 6.404));

  // Check if the number of steps performed is less or equal as the maximum
  // allowed, since we use a random action there is no guarantee that we will
  // reach the maximum number of steps.
  BOOST_REQUIRE_LE(task.StepsPerformed(), 20);

  // The action is simply the torque. Check if dimension is 1.
  BOOST_REQUIRE_EQUAL(1, static_cast<size_t>(Pendulum::Action::size));
}

/**
 * Constructs a Continuous MountainCar instance and check if the main rountine
 * works as it should be.
 */
BOOST_AUTO_TEST_CASE(SimpleContinuousMountainCarTest)
{
  ContinuousMountainCar task = ContinuousMountainCar();
  task.MaxSteps() = 5;

  ContinuousMountainCar::State state = task.InitialSample();
  ContinuousMountainCar::Action action;
  action.action[0] = math::Random(-1.0, 1.0);
  double reward = task.Sample(state, action);
  // Maximum reward possible is 100.
  BOOST_REQUIRE(reward <= 100.0);
  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
    task.Sample(state, action, state);

  // Check if the number of steps performed is the same as the maximum allowed.
  BOOST_REQUIRE_EQUAL(task.StepsPerformed(), 5);

  // Check if the size of the action space is 1.
  BOOST_REQUIRE_EQUAL(1, action.size);
}

/**
 * Constructs a Acrobot instance and check if the main rountine works as
 * it should be.
 */
BOOST_AUTO_TEST_CASE(SimpleAcrobotTest)
{
  Acrobot task = Acrobot();
  task.MaxSteps() = 5;

  Acrobot::State state = task.InitialSample();
  Acrobot::Action action;
  action.action = Acrobot::Action::actions::negativeTorque;
  double reward = task.Sample(state, action);

  BOOST_REQUIRE_EQUAL(reward, -1.0);
  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
    task.Sample(state, action, state);

  // Check if the number of steps performed is the same as the maximum allowed.
  BOOST_REQUIRE_EQUAL(task.StepsPerformed(), 5);

  // Check if the size of the action space is 3.
  BOOST_REQUIRE_EQUAL(3, static_cast<size_t>(Acrobot::Action::size));
}

/**
 * Constructs a MountainCar instance and check if the main rountine works as
 * it should be.
 */
BOOST_AUTO_TEST_CASE(SimpleMountainCarTest)
{
  MountainCar task = MountainCar();
  task.MaxSteps() = 5;

  MountainCar::State state = task.InitialSample();
  MountainCar::Action action;
  action.action = MountainCar::Action::actions::backward;
  double reward = task.Sample(state, action);

  BOOST_REQUIRE_EQUAL(reward, -1.0);
  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
    task.Sample(state, action, state);

  // Check if the number of steps performed is the same as the maximum allowed.
  BOOST_REQUIRE_EQUAL(task.StepsPerformed(), 5);

  // Check if the size of the action space is 3.
  BOOST_REQUIRE_EQUAL(3, static_cast<size_t>(MountainCar::Action::size));
}

/**
 * Constructs a CartPole instance and check if the main routine works as
 * it should be.
 */
BOOST_AUTO_TEST_CASE(SimpleCartPoleTest)
{
  CartPole task = CartPole();
  task.MaxSteps() = 5;

  CartPole::State state = task.InitialSample();
  CartPole::Action action;
  action.action = CartPole::Action::actions::backward;
  double reward = task.Sample(state, action);

  BOOST_REQUIRE_EQUAL(reward, 1.0);
  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
    task.Sample(state, action, state);

  // Check if the number of steps performed is the same as the maximum allowed.
  BOOST_REQUIRE_EQUAL(task.StepsPerformed(), 5);

  BOOST_REQUIRE_EQUAL(2, static_cast<size_t>(CartPole::Action::size));
}

/**
 * Constructs a DoublePoleCart instance and check if the main routine works as
 * it should be.
 */
BOOST_AUTO_TEST_CASE(DoublePoleCartTest)
{
  DoublePoleCart task = DoublePoleCart();
  task.MaxSteps() = 5;

  DoublePoleCart::State state = task.InitialSample();
  DoublePoleCart::Action action;
  action.action = DoublePoleCart::Action::actions::backward;
  double reward = task.Sample(state, action);

  BOOST_REQUIRE_EQUAL(reward, 1.0);
  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
    task.Sample(state, action, state);

  // Check if the number of steps performed is the same as the maximum allowed.
  BOOST_REQUIRE_EQUAL(task.StepsPerformed(), 5);
  BOOST_REQUIRE_EQUAL(2, static_cast<size_t>(DoublePoleCart::Action::size));
}

/**
 * Constructs a ContinuousDoublePoleCart instance and check if the main 
 * routine works as it should be.
 */
BOOST_AUTO_TEST_CASE(ContinuousDoublePoleCartTest)
{
  ContinuousDoublePoleCart task = ContinuousDoublePoleCart();
  task.MaxSteps() = 5;

  ContinuousDoublePoleCart::State state = task.InitialSample();
  ContinuousDoublePoleCart::Action action;
  action.action[0] = math::Random(-1.0, 1.0);
  double reward = task.Sample(state, action);

  BOOST_REQUIRE_EQUAL(reward, 1.0);
  BOOST_REQUIRE(!task.IsTerminal(state));

  while (!task.IsTerminal(state))
    task.Sample(state, action, state);

  // Check if the number of steps performed is the same as the maximum allowed.
  BOOST_REQUIRE_EQUAL(task.StepsPerformed(), 5);
  BOOST_REQUIRE_EQUAL(1, action.size);
}

/**
 * Construct a random replay instance and check if it works as
 * it should be.
 */
BOOST_AUTO_TEST_CASE(RandomReplayTest)
{
  RandomReplay<MountainCar> replay(1, 3);
  MountainCar env;
  MountainCar::State state = env.InitialSample();
  MountainCar::Action action;
  action.action = MountainCar::Action::actions::forward;
  MountainCar::State nextState;
  double reward = env.Sample(state, action, nextState);
  replay.Store(state, action, reward, nextState, env.IsTerminal(nextState),
      0.9);
  arma::mat sampledState;
  std::vector<MountainCar::Action> sampledAction;
  arma::rowvec sampledReward;
  arma::mat sampledNextState;
  arma::irowvec sampledTerminal;

  //! So far there should be only one record in the memory
  replay.Sample(sampledState, sampledAction, sampledReward, sampledNextState,
      sampledTerminal);

  CheckMatrices(state.Encode(), sampledState);
  BOOST_REQUIRE_EQUAL(sampledAction.size(), 1);
  BOOST_REQUIRE_EQUAL(action.action, sampledAction[0].action);
  BOOST_REQUIRE_CLOSE(reward, arma::as_scalar(sampledReward), 1e-5);
  CheckMatrices(nextState.Encode(), sampledNextState);
  BOOST_REQUIRE_EQUAL(false, arma::as_scalar(sampledTerminal));
  BOOST_REQUIRE_EQUAL(1, replay.Size());

  //! Overwrite the memory with a nonsense record
  for (size_t i = 0; i < 5; ++i)
    replay.Store(nextState, action, reward, state, true, 0.9);

  BOOST_REQUIRE_EQUAL(3, replay.Size());

  //! Sample several times, the original record shouldn't appear
  for (size_t i = 0; i < 30; ++i)
  {
    replay.Sample(sampledState, sampledAction, sampledReward, sampledNextState,
        sampledTerminal);

    CheckMatrices(state.Encode(), sampledNextState);
    CheckMatrices(nextState.Encode(), sampledState);
    BOOST_REQUIRE_EQUAL(true, arma::as_scalar(sampledTerminal));
  }
}

/**
 * Construct a greedy policy instance and check if it works as
 * it should be.
 */
BOOST_AUTO_TEST_CASE(GreedyPolicyTest)
{
  GreedyPolicy<CartPole> policy(1.0, 10, 0.0, 0.99);
  for (size_t i = 0; i < 15; ++i)
    policy.Anneal();
  BOOST_REQUIRE_CLOSE(0.0, policy.Epsilon(), 1e-5);
  arma::colvec actionValue = arma::randn<arma::colvec>(CartPole::Action::size);
  CartPole::Action action = policy.Sample(actionValue);
  BOOST_REQUIRE_CLOSE(actionValue[action.action], actionValue.max(), 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
