
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_reward_clipping_test.cpp:

Program Listing for File reward_clipping_test.cpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_reward_clipping_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/reward_clipping_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   
   #include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
   #include <mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp>
   #include <mlpack/methods/reinforcement_learning/environment/continuous_mountain_car.hpp>
   #include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
   #include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
   #include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>
   #include <mlpack/methods/reinforcement_learning/environment/reward_clipping.hpp>
   
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
   #include <mlpack/methods/reinforcement_learning/q_learning.hpp>
   #include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
   #include <mlpack/methods/reinforcement_learning/training_config.hpp>
   
   #include <ensmallen.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace mlpack::ann;
   using namespace ens;
   using namespace mlpack::rl;
   
   
   // Test checking that reward clipping works with vanilla update.
   TEST_CASE("ClippedRewardTest", "[RewardClippingTest]")
   {
     Pendulum task;
     RewardClipping<Pendulum> rewardClipping(task, -2.0, +2.0);
   
     RewardClipping<Pendulum>::State state = rewardClipping.InitialSample();
     RewardClipping<Pendulum>::Action action;
     action.action[0] = mlpack::math::Random(-1.0, 1.0);
     double reward = rewardClipping.Sample(state, action);
   
     REQUIRE(reward <= 2.0);
     REQUIRE(reward >= -2.0);
   }
   
   TEST_CASE("RewardClippedAcrobotWithDQN", "[RewardClippingTest]")
   {
     // We will allow three trials, although it would be very uncommon for the test
     // to use more than one.
     bool converged = false;
     for (size_t trial = 0; trial < 3; ++trial)
     {
       // Set up the network.
       SimpleDQN<> model(4, 64, 32, 3);
   
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
