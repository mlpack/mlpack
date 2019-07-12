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

// Activation functions.
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <mlpack/tests/neat_test_tools.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;
using namespace mlpack::ann;
using namespace mlpack::neat;

BOOST_AUTO_TEST_SUITE(NEATTest)

/**
 * Test NEAT on the XOR Test.
 */
BOOST_AUTO_TEST_CASE(NEATXORTest)
{
  XORTask task;
  NEAT<XORTask> model(task, 2, 1, 100, 150, 10);
  model.FinalFitness() = 3.9;

  // Find the best genome and it's fitness.
  Genome<> bestGenome = model.Train();
  double finalFitness = bestGenome.Fitness();

  // Check if the final fitness is acceptable.
  BOOST_REQUIRE(finalFitness >= 3.9);
}

/**
 * Test NEAT on the Double Pole Cart Balancing environment without velocities.
 */
BOOST_AUTO_TEST_CASE(NEATDoublePoleCartNoVelocitiesTest)
{
  arma::vec poleLengths = {0.5, 0.05};
  arma::vec poleMasses = {0.1, 0.01};
  const MultiplePoleCart env = MultiplePoleCart(2, poleLengths, poleMasses,
      9.8, 1, 0.01);
  DPNVTask task(env);
  NEAT<DPNVTask> model(task, 3, 1, 1000, 200, 50, 0, 1, 0.8, 1.8, 0.5, 0.01,
      0.3, 0.2, 0.05, 0);

  // Find the best genome and it's fitness.
  Genome<> bestGenome = model.Train();
  double finalFitness = bestGenome.Fitness();
  std::cout << finalFitness << std::endl;

  // Check if the final fitness is acceptable (Placeholder).
  BOOST_REQUIRE(finalFitness >= 300);
}

BOOST_AUTO_TEST_SUITE_END()
