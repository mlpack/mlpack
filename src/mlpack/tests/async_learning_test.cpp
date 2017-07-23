/**
 * @file async_learning_test.hpp
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

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/reinforcement_learning/async_learning.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/core/optimizers/adam/adam_update.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/policy/aggregated_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace mlpack::rl;

BOOST_AUTO_TEST_SUITE(AsyncLearningTest);

// Test async one step q-learning in Cart Pole.
BOOST_AUTO_TEST_CASE(OneStepQLearningTest)
{
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> model(MeanSquaredError<>(),
      GaussianInitialization(0, 0.001));
  model.Add<Linear<>>(4, 20);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(20, 20);
  model.Add<ReLULayer<>>();
  model.Add<Linear<>>(20, 2);

  /**
   * Load the pre-trained network.
   * Notice that even you train the network from scratch, it should also be
   * pretty fast in a modern laptop (e.g. 2s in a macbook pro 2016).
   * Here we load the pre-trained network mainly for the mlpack test server.
   */
  std::string fileName("async_one_step_q_learning_network.bin");
  bool loadNetwork = true;
  bool storeNetwork = false;
  if (loadNetwork)
    data::Load(fileName, "network", model);

  // Set up the policy.
  using Policy = GreedyPolicy<CartPole>;
  AggregatedPolicy<Policy> policy({Policy(0.7, 5000, 0.1),
                                  Policy(0.7, 5000, 0.01),
                                  Policy(0.7, 5000, 0.5)},
                                  arma::colvec("0.4 0.3 0.3"));

  TrainingConfig config;
  config.StepSize() = 0.0001;
  config.Discount() = 0.99;
  /**
   * When training from scratch, you should set proper number of
   * workers (e.g. 16).
   */
  config.NumOfWorkers() = 1;
  config.UpdateInterval() = 6;
  config.StepLimit() = 200;
  config.TargetNetworkSyncInterval() = 200;

  OneStepQLearning<CartPole, decltype(model), VanillaUpdate, decltype(policy)>
      agent(std::move(config), std::move(model), std::move(policy));

  arma::vec rewards(20, arma::fill::zeros);
  size_t pos = 0;
  size_t testEpisodes = 0;
  auto measure = [&rewards, &pos, &testEpisodes](double reward) {
    size_t maxEpisode = 100000;
    if (testEpisodes > maxEpisode)
      BOOST_REQUIRE(false);
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

  // Store the trained network.
  if (storeNetwork)
    data::Save(fileName, "network", agent.Network(), false);
}

BOOST_AUTO_TEST_SUITE_END();
