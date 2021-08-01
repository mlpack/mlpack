
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_rl_components_test.cpp:

Program Listing for File rl_components_test.cpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_rl_components_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/rl_components_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
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
   
   #include "catch.hpp"
   #include "test_catch_tools.hpp"
   
   using namespace mlpack;
   using namespace mlpack::rl;
   
   TEST_CASE("SimplePendulumTest", "[RLComponentsTest]")
   {
     Pendulum task = Pendulum();
     task.MaxSteps() = 20;
   
     Pendulum::State state = task.InitialSample();
     Pendulum::Action action;
     action.action[0] = math::Random(-2.0, 2.0);
     double reward, minReward = 0.0;
   
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
     {
       reward = task.Sample(state, action, state);
       minReward = std::min(reward, minReward);
     }
   
     // The reward is always negative. Check if not lower than lowest possible.
     REQUIRE(minReward >= -(pow(M_PI, 2) + 6.404));
   
     // Check if the number of steps performed is less or equal as the maximum
     // allowed, since we use a random action there is no guarantee that we will
     // reach the maximum number of steps.
     REQUIRE(task.StepsPerformed() <= 20);
   
     // The action is simply the torque. Check if dimension is 1.
     REQUIRE(1 == static_cast<size_t>(Pendulum::Action::size));
   }
   
   TEST_CASE("SimpleContinuousMountainCarTest", "[RLComponentsTest]")
   {
     ContinuousMountainCar task = ContinuousMountainCar();
     task.MaxSteps() = 5;
   
     ContinuousMountainCar::State state = task.InitialSample();
     ContinuousMountainCar::Action action;
     action.action[0] = math::Random(-1.0, 1.0);
     double reward = task.Sample(state, action);
     // Maximum reward possible is 100.
     REQUIRE(reward <= 100.0);
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
       task.Sample(state, action, state);
   
     // Check if the number of steps performed is the same as the maximum allowed.
     REQUIRE(task.StepsPerformed() == 5);
   
     // Check if the size of the action space is 1.
     REQUIRE(1 == action.size);
   }
   
   TEST_CASE("SimpleAcrobotTest", "[RLComponentsTest]")
   {
     Acrobot task = Acrobot();
     task.MaxSteps() = 5;
   
     Acrobot::State state = task.InitialSample();
     Acrobot::Action action;
     action.action = Acrobot::Action::actions::negativeTorque;
     double reward = task.Sample(state, action);
   
     REQUIRE(reward == -1.0);
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
       task.Sample(state, action, state);
   
     // Check if the number of steps performed is the same as the maximum allowed.
     REQUIRE(task.StepsPerformed() == 5);
   
     // Check if the size of the action space is 3.
     REQUIRE(3 == static_cast<size_t>(Acrobot::Action::size));
   }
   
   TEST_CASE("SimpleMountainCarTest", "[RLComponentsTest]")
   {
     MountainCar task = MountainCar();
     task.MaxSteps() = 5;
   
     MountainCar::State state = task.InitialSample();
     MountainCar::Action action;
     action.action = MountainCar::Action::actions::backward;
     double reward = task.Sample(state, action);
   
     REQUIRE(reward == -1.0);
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
       task.Sample(state, action, state);
   
     // Check if the number of steps performed is the same as the maximum allowed.
     REQUIRE(task.StepsPerformed() == 5);
   
     // Check if the size of the action space is 3.
     REQUIRE(3 == static_cast<size_t>(MountainCar::Action::size));
   }
   
   TEST_CASE("SimpleCartPoleTest", "[RLComponentsTest]")
   {
     CartPole task = CartPole();
     task.MaxSteps() = 5;
   
     CartPole::State state = task.InitialSample();
     CartPole::Action action;
     action.action = CartPole::Action::actions::backward;
     double reward = task.Sample(state, action);
   
     REQUIRE(reward == 1.0);
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
       task.Sample(state, action, state);
   
     // Check if the number of steps performed is the same as the maximum allowed.
     REQUIRE(task.StepsPerformed() == 5);
   
     REQUIRE(2 == static_cast<size_t>(CartPole::Action::size));
   }
   
   TEST_CASE("DoublePoleCartTest", "[RLComponentsTest]")
   {
     DoublePoleCart task = DoublePoleCart();
     task.MaxSteps() = 5;
   
     DoublePoleCart::State state = task.InitialSample();
     DoublePoleCart::Action action;
     action.action = DoublePoleCart::Action::actions::backward;
     double reward = task.Sample(state, action);
   
     REQUIRE(reward == 1.0);
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
       task.Sample(state, action, state);
   
     // Check if the number of steps performed is the same as the maximum allowed.
     REQUIRE(task.StepsPerformed() == 5);
     REQUIRE(2 == static_cast<size_t>(DoublePoleCart::Action::size));
   }
   
   TEST_CASE("ContinuousDoublePoleCartTest", "[RLComponentsTest]")
   {
     ContinuousDoublePoleCart task = ContinuousDoublePoleCart();
     task.MaxSteps() = 5;
   
     ContinuousDoublePoleCart::State state = task.InitialSample();
     ContinuousDoublePoleCart::Action action;
     action.action[0] = math::Random(-1.0, 1.0);
     double reward = task.Sample(state, action);
   
     REQUIRE(reward == 1.0);
     REQUIRE(!task.IsTerminal(state));
   
     while (!task.IsTerminal(state))
       task.Sample(state, action, state);
   
     // Check if the number of steps performed is the same as the maximum allowed.
     REQUIRE(task.StepsPerformed() == 5);
     REQUIRE(1 == action.size);
   }
   
   TEST_CASE("RandomReplayTest", "[RLComponentsTest]")
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
   
     replay.Sample(sampledState, sampledAction, sampledReward, sampledNextState,
         sampledTerminal);
   
     CheckMatrices(state.Encode(), sampledState);
     REQUIRE(sampledAction.size() == 1);
     REQUIRE(action.action == sampledAction[0].action);
     REQUIRE(reward == Approx(arma::as_scalar(sampledReward)).epsilon(1e-7));
     CheckMatrices(nextState.Encode(), sampledNextState);
     REQUIRE(false == arma::as_scalar(sampledTerminal));
     REQUIRE(1 == replay.Size());
   
     for (size_t i = 0; i < 5; ++i)
       replay.Store(nextState, action, reward, state, true, 0.9);
   
     REQUIRE(3 == replay.Size());
   
     for (size_t i = 0; i < 30; ++i)
     {
       replay.Sample(sampledState, sampledAction, sampledReward, sampledNextState,
           sampledTerminal);
   
       CheckMatrices(state.Encode(), sampledNextState);
       CheckMatrices(nextState.Encode(), sampledState);
       REQUIRE(true == arma::as_scalar(sampledTerminal));
     }
   }
   
   TEST_CASE("GreedyPolicyTest", "[RLComponentsTest]")
   {
     GreedyPolicy<CartPole> policy(1.0, 10, 0.0, 0.99);
     for (size_t i = 0; i < 15; ++i)
       policy.Anneal();
     REQUIRE(0.0 == Approx(policy.Epsilon()).epsilon(1e-7));
     arma::colvec actionValue = arma::randn<arma::colvec>(CartPole::Action::size);
     CartPole::Action action = policy.Sample(actionValue);
     REQUIRE(actionValue[action.action] ==
         Approx(actionValue.max()).epsilon(1e-7));
   }
