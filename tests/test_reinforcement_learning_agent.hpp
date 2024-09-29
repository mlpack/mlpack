/**
 * @file tests/test_reinforcement_learning_agent.hpp
 * @author Tarek Elsayed
 * 
 * Helper functions for reinforcement learning agents testing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_TEST_REINFORCEMENT_LEARNING_AGENT_HPP
#define MLPACK_TESTS_TEST_REINFORCEMENT_LEARNING_AGENT_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>
#include <mlpack/methods/reinforcement_learning.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace ens;

/**
 * Test the reinforcement learning agent's performance and 
 * check if it converges to the desired reward level.
 */
template<typename AgentType>
bool testAgent(AgentType& agent,
               const double rewardThreshold,
               const size_t noOfEpisodes,
               const size_t consecutiveEpisodesTest = 50)
{
  bool converged = false;
  std::vector<double> returnList;
  size_t episodes = 0;
  while (true)
  {
    double episodeReturn = agent.Episode();
    episodes += 1;
    returnList.push_back(episodeReturn);

    if (returnList.size() > consecutiveEpisodesTest)
      returnList.erase(returnList.begin());

    double averageReturn = std::accumulate(returnList.begin(),
        returnList.end(), 0.0) / returnList.size();

    Log::Debug << "Average return in last " << returnList.size()
        << " consecutive episodes: " << averageReturn
        << " Episode return: " << episodeReturn << std::endl;

    // For the speed of the test case, a high criterion should not be set
    // for the rewardThreshold.
    if (averageReturn > rewardThreshold &&
        returnList.size() >= consecutiveEpisodesTest)
    {
      converged = true;
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      Log::Debug << "Average return in deterministic test: "
          << testReturn.mean() << std::endl;
      break;
    }

    if (episodes > noOfEpisodes)
    {
      Log::Debug << "Agent failed." << std::endl;
      break;
    }
  }

  return converged;
}

#endif
