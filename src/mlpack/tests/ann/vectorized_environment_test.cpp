/**
 * @file tests/vectorized_environment_test.cpp
 * @author Ali Hossam
 *
 * Test for the Vectorized Environment wrapper for reinforcement learning 
 * environments.
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

template<>
size_t VecEnv<Pendulum>::nEnvs = 4;

// Test SAC in Pendulum Environment.
TEST_CASE("VecEnvTestSACPendulum", "[VecEnvTest]")
{

  // Set up the replay method.
  RandomReplay<Pendulum> replayMethod(32, 10000);

  // Set up Vectorized environment replay wrapper.
  using ReplayType = VecEnvReplay<Pendulum, RandomReplay<Pendulum>>; 
  ReplayType vecReplay(replayMethod);
  
  
  TrainingConfig config;
  config.StepSize() = 0.01;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;
  config.Rho() = 0.001;


  FFN<EmptyLoss, GaussianInitialization>
      policyNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear(128));
  policyNetwork.Add(new ReLU());
  policyNetwork.Add(new Linear(1));
  policyNetwork.Add(new TanH());

  FFN<EmptyLoss, GaussianInitialization>
      qNetwork(EmptyLoss(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear(128));
  qNetwork.Add(new ReLU());
  qNetwork.Add(new Linear(1));


  // Set up Soft actor-critic agent.
  SAC<VecEnv<Pendulum>, decltype(qNetwork), decltype(policyNetwork),
     AdamUpdate, ReplayType> agent(config, qNetwork, policyNetwork, vecReplay);

  // Set the thresholds for testing.
  const double rewardThreshold = -300;
  const size_t noOfEpisodes = 1000;
  const size_t consecutiveEpisodesTest = 10;

  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = false;
  while (true)
  {
    double episodeReturn = agent.Episode();
    episodes += 1;
    returnList.push_back(episodeReturn);

    if (returnList.size() > consecutiveEpisodesTest)
      returnList.erase(returnList.begin());

    double averageReturn = std::accumulate(returnList.begin(),
        returnList.end(), 0.0) / returnList.size();

    Log::Debug << "Episode Number " << episodes 
        << " Average return in last " << returnList.size()
        << " consecutive episodes: " << averageReturn
        << " Episode return: " << episodeReturn << std::endl;

    if (averageReturn > rewardThreshold &&
        returnList.size() >= consecutiveEpisodesTest)
    {
      // Test the agent for 10 episodes.
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Agent successfully converged.\n"
          << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      
      converged = true;
      break;
    }

    if (episodes > noOfEpisodes)
    {
      Log::Debug << "Agent failed to converge." << std::endl;
      break;
    }
  }

  REQUIRE(converged);

}
