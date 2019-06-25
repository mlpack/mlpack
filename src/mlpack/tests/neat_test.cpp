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

// RL environments.
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/continuous_mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/environment/continuous_multiple_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>

// Activation functions.
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>

#include <mlpack/tests/neat_test_tools.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <chrono>

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
  arma::arma_rng::set_seed_random();
  auto t1 = std::chrono::high_resolution_clock::now();
  XORTask task;
  NEAT<XORTask, HardSigmoidFunction, RankSelection> model(task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

  // Find the best genome and it's fitness.
  Genome<HardSigmoidFunction> bestGenome = model.Train();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "NEAT took "
              << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count()
              << " seconds\n";
  bestGenome.Parameters().print();
  double finalFitness = bestGenome.Fitness();
  std::cout << finalFitness << std::endl;
  // Check if the final fitness is acceptable.
  BOOST_REQUIRE(finalFitness >= 3.5);
}

/**
 * Test NEAT on the Pendulum environment.
 */
// BOOST_AUTO_TEST_CASE(NEATPendulumTest)
// {
//   const Pendulum env = Pendulum();
//   ContinuousRLTask<Pendulum> task(env);
//   NEAT<ContinuousRLTask<Pendulum>, HardSigmoidFunction, RankSelection> model
//       (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = bestGenome.Fitness();
//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

/**
 * Test NEAT on the ContinuousMountainCar environment.
 */
BOOST_AUTO_TEST_CASE(NEATContinuousMountainCarTest)
{
  const ContinuousMountainCar env = ContinuousMountainCar();
  ContinuousRLTask<ContinuousMountainCar> task(env);
  NEAT<ContinuousRLTask<ContinuousMountainCar>, HardSigmoidFunction, RankSelection> model
      (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

  // Find the best genome and it's fitness.
  Genome<HardSigmoidFunction> bestGenome = model.Train();
  double finalFitness = bestGenome.Fitness();
  std::cout << finalFitness << std::endl;

  // Check if the final fitness is acceptable (Placeholder).
  BOOST_REQUIRE(finalFitness >= 90);
}

/**
 * Test NEAT on the MountainCar environment.
 */
BOOST_AUTO_TEST_CASE(NEATMountainCarTest)
{
  const MountainCar env = MountainCar();
  DiscreteRLTask<MountainCar> task(env);
  NEAT<DiscreteRLTask<MountainCar>, HardSigmoidFunction, RankSelection> model
      (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

  // Find the best genome and it's fitness.
  Genome<HardSigmoidFunction> bestGenome = model.Train();
  double finalFitness = task.Evaluate(bestGenome);
  std::cout << finalFitness << std::endl;

  // Check if the final fitness is acceptable (Placeholder).
  BOOST_REQUIRE(finalFitness >= 90);
}

/**
 * Test NEAT on the Acrobot environment.
 */
BOOST_AUTO_TEST_CASE(NEATAcrobotTest)
{
  const Acrobot env = Acrobot();
  DiscreteRLTask<Acrobot> task(env);
  NEAT<DiscreteRLTask<Acrobot>, HardSigmoidFunction, RankSelection> model
      (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

  // Find the best genome and it's fitness.
  Genome<HardSigmoidFunction> bestGenome = model.Train();
  double finalFitness = task.Evaluate(bestGenome);
  std::cout << finalFitness << std::endl;

  // Check if the final fitness is acceptable (Placeholder).
  BOOST_REQUIRE(finalFitness >= 90);
}

/**
 * Test NEAT on the CartPole environment.
 */
BOOST_AUTO_TEST_CASE(NEATCartPoleTest)
{
  const CartPole env = CartPole();
  DiscreteRLTask<CartPole> task(env);
  NEAT<DiscreteRLTask<CartPole>, HardSigmoidFunction, RankSelection> model
       (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

  // Find the best genome and it's fitness.
  Genome<HardSigmoidFunction> bestGenome = model.Train();
  double finalFitness = task.Evaluate(bestGenome);
  std::cout << finalFitness << std::endl;

  // Check if the final fitness is acceptable (Placeholder).
  BOOST_REQUIRE(finalFitness >= 90);
}

// /**
//  * Test NEAT on the Double Pole Cart Balancing environment.
//  */
// BOOST_AUTO_TEST_CASE(NEATDoublePoleCartTest)
// {
//   arma::vec poleLengths = {1, 0.1};
//   arma::vec poleMasses = {1, 0.1};
//   const ContinuousMultiplePoleCart env = ContinuousMultiplePoleCart(2,
//       poleLengths, poleMasses);
//   DPVTask task(env);
//   NEAT<DPVTask> model(task, 6 /* Input nodes */, 1 /* Output nodes */);

//   // Find the best genome and it's fitness.
//   Genome bestGenome = model.Train();
//   double finalFitness = task.Evaluate(bestGenome);

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

// /**
//  * Test NEAT on the Double Pole Cart Balancing environment without velocities.
//  */
// BOOST_AUTO_TEST_CASE(NEATDoublePoleCartNoVelocitiesTest)
// {
//   arma::vec poleLengths = {1, 0.1};
//   arma::vec poleMasses = {1, 0.1};
//   const ContinuousMultiplePoleCart env = ContinuousMultiplePoleCart(2,
//       poleLengths, poleMasses);
//   DPNVTask task(env);
//   NEAT<DPNVTask> model(task, 3 /* Input nodes */, 1 /* Output nodes */);

//   // Find the best genome and it's fitness.
//   Genome bestGenome = model.Train();
//   double finalFitness = task.Evaluate(bestGenome);

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

BOOST_AUTO_TEST_SUITE_END()
