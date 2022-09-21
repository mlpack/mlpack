/**
 * @file tests/async_learning_test.cpp
 * @author Shangtong Zhang
 *
 * Test for async deep RL methods.
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

// Test async one step q-learning in Cart Pole.
TEST_CASE("OneStepQLearningTest", "[AsyncLearningTest]")
{
  /**
   * This is for the Travis CI server, in your own machine you should use more
   * threads.
   */
  #ifdef MLPACK_USE_OPENMP
    omp_set_num_threads(1);
  #endif

  bool success = false;
  for (size_t trial = 0; trial < 4; ++trial)
  {
    // Set up the network.
    FFN<MeanSquaredError, GaussianInitialization> model(MeanSquaredError(),
        GaussianInitialization(0, 0.001));
    model.Add<Linear>(20);
    model.Add<ReLU>();
    model.Add<Linear>(20);
    model.Add<ReLU>();
    model.Add<Linear>(2);

    // Set up the policy.
    using Policy = GreedyPolicy<CartPole>;
    AggregatedPolicy<Policy> policy({Policy(0.7, 5000, 0.1),
                                     Policy(0.7, 5000, 0.01),
                                     Policy(0.7, 5000, 0.5)},
                                     arma::colvec("0.4 0.3 0.3"));

    TrainingConfig config;
    config.StepSize() = 0.0001;
    config.Discount() = 0.99;
    config.NumWorkers() = 16;
    config.UpdateInterval() = 6;
    config.StepLimit() = 200;
    config.TargetNetworkSyncInterval() = 200;

    OneStepQLearning<
        CartPole, decltype(model), ens::VanillaUpdate, decltype(policy)>
        agent(std::move(config), std::move(model), std::move(policy));

    arma::vec rewards(20, arma::fill::zeros);
    size_t pos = 0;
    size_t testEpisodes = 0;
    auto measure = [&rewards, &pos, &testEpisodes](double reward)
    {
      size_t maxEpisode = 10000;
      if (testEpisodes > maxEpisode)
        return true; // Fake convergence...
      testEpisodes++;
      rewards[pos++] = reward;
      pos %= rewards.n_elem;
      // Maybe underestimated.
      double avgReward = arma::mean(rewards);
      Log::Debug << "Average return: " << avgReward
          << " Episode return: " << reward << std::endl;
      if (avgReward > 60)
        return true;
      return false;
    };

    agent.Train(measure);
    Log::Debug << "Total test episodes: " << testEpisodes << std::endl;

    double avgReward = arma::mean(rewards);
    if (avgReward > 60)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

// Test async one step Sarsa in Cart Pole.
TEST_CASE("OneStepSarsaTest", "[AsyncLearningTest]")
{
  /**
   * This is for the Travis CI server, in your own machine you shuold use more
   * threads.
   */
  #ifdef MLPACK_USE_OPENMP
    omp_set_num_threads(1);
  #endif

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    // Set up the network.
    FFN<MeanSquaredError, GaussianInitialization> model(MeanSquaredError(),
        GaussianInitialization(0, 0.001));
    model.Add<Linear>(20);
    model.Add<ReLU>();
    model.Add<Linear>(20);
    model.Add<ReLU>();
    model.Add<Linear>(2);

    // Set up the policy.
    using Policy = GreedyPolicy<CartPole>;
    AggregatedPolicy<Policy> policy({Policy(0.7, 5000, 0.1),
                                     Policy(0.7, 5000, 0.01),
                                     Policy(0.7, 5000, 0.5)},
                                     arma::colvec("0.4 0.3 0.3"));

    TrainingConfig config;
    config.StepSize() = 0.0001;
    config.Discount() = 0.99;
    config.NumWorkers() = 16;
    config.UpdateInterval() = 6;
    config.StepLimit() = 200;
    config.TargetNetworkSyncInterval() = 200;

    OneStepSarsa<CartPole,
                 decltype(model),
                 ens::VanillaUpdate,
                 decltype(policy)>
    agent(std::move(config), std::move(model), std::move(policy));

    arma::vec rewards(20, arma::fill::zeros);
    size_t pos = 0;
    size_t testEpisodes = 0;
    auto measure = [&rewards, &pos, &testEpisodes](double reward)
    {
      size_t maxEpisode = 10000;
      if (testEpisodes > maxEpisode)
        return true; // Fake convergence...
      testEpisodes++;
      rewards[pos++] = reward;
      pos %= rewards.n_elem;
      // Maybe underestimated.
      double avgReward = arma::mean(rewards);
      Log::Debug << "Average return: " << avgReward
                 << " Episode return: " << reward << std::endl;
      if (avgReward > 60)
        return true;
      return false;
    };

    agent.Train(measure);
    Log::Debug << "Total test episodes: " << testEpisodes << std::endl;

    double avgReward = arma::mean(rewards);
    if (avgReward > 60)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

// Test async n step q-learning in Cart Pole.
TEST_CASE("NStepQLearningTest", "[AsyncLearningTest]")
{
  /**
   * This is for the Travis CI server, in your own machine you shuold use more
   * threads.
   */
  #ifdef MLPACK_USE_OPENMP
    omp_set_num_threads(1);
  #endif

  // Set up the network.
  FFN<MeanSquaredError, GaussianInitialization> model(MeanSquaredError(),
      GaussianInitialization(0, 0.001));
  model.Add<Linear>(20);
  model.Add<ReLU>();
  model.Add<Linear>(20);
  model.Add<ReLU>();
  model.Add<Linear>(2);

  // Set up the policy.
  using Policy = GreedyPolicy<CartPole>;
  AggregatedPolicy<Policy> policy({Policy(0.7, 5000, 0.1),
                                   Policy(0.7, 5000, 0.01),
                                   Policy(0.7, 5000, 0.5)},
                                  arma::colvec("0.4 0.3 0.3"));

  TrainingConfig config;
  config.StepSize() = 0.0001;
  config.Discount() = 0.99;
  config.NumWorkers() = 16;
  config.UpdateInterval() = 6;
  config.StepLimit() = 200;
  config.TargetNetworkSyncInterval() = 200;

  NStepQLearning<
      CartPole, decltype(model), ens::VanillaUpdate, decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy));

  arma::vec rewards(20, arma::fill::zeros);
  size_t pos = 0;
  size_t testEpisodes = 0;
  auto measure = [&rewards, &pos, &testEpisodes](double reward)
  {
    size_t maxEpisode = 100000;
    if (testEpisodes > maxEpisode)
      REQUIRE(false);
    testEpisodes++;
    rewards[pos++] = reward;
    pos %= rewards.n_elem;
    // Maybe underestimated.
    double avgReward = arma::mean(rewards);
    Log::Debug << "Average return: " << avgReward
               << " Episode return: " << reward << std::endl;
    if (avgReward > 60)
      return true;
    return false;
  };

  agent.Train(measure);
  Log::Debug << "Total test episodes: " << testEpisodes << std::endl;
}
