/**
 * @file neat_test.cpp
 * @author Rahul Ganesh Prabhu
 *
 * Tests for NeuroEvolution of Augmenting Topologies.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>

// NEAT resources.
#include <mlpack/methods/neat/neat.hpp>
#include <mlpack/methods/neat/selection_strategies/rank_selection.hpp>

#include <mlpack/tests/neat_test_tools.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;
using namespace mlpack::ann;
using namespace mlpack::neat;

BOOST_AUTO_TEST_SUITE(NEATTest)

/**
 * Test NEAT and phased searching on the XOR Test. XOR normally performs better
 *  in cyclic cases, this is meant to be a test of the acyclic network.
 */
BOOST_AUTO_TEST_CASE(NEATXORTest)
{
  XORTask task;
  NEAT<XORTask> model(task, 2, 1, 100, 500, 10);
  model.FinalFitness() = 3.8;
  model.IsAcyclic() = true;
  model.ComplexityThreshold() = 6;

  // Find the best genome and it's fitness.
  Genome<> bestGenome = model.Train();
  double finalFitness = bestGenome.Fitness();
  double meanComplexity = model.MeanComplexity();

  // Check if the final fitness is acceptable.
  BOOST_REQUIRE(finalFitness >= 3.8);

  // Check if the complexity is not too far in excess of the complexity
  // ceiling.
  BOOST_REQUIRE(meanComplexity < (double)model.CurrentComplexityCeiling() + 1);
}

/**
 * Test NEAT on the Double Pole Cart Balancing environment without velocities.
 */
BOOST_AUTO_TEST_CASE(NEATDoublePoleCartNoVelocitiesTest)
{
  MultiplePoleCart env = MultiplePoleCart();
  DPNVTask task(env);
  NEAT<DPNVTask> model(task, 3, 1, 1000, 200, 50, 0, 1, 0.8, 1.8, 0.5, 0.01,
      0.3, 0.2, 0.05, 0);
  model.FinalFitness() = 10000;

  // Find the best genome and it's fitness.
  Genome<> bestGenome = model.Train();
  double finalFitness = bestGenome.Fitness();
  Log::Debug << "The final fitness is " << finalFitness << std::endl;

  // Check if the final fitness is acceptable.
  BOOST_REQUIRE(finalFitness >= 10000);
}

BOOST_AUTO_TEST_SUITE_END()
