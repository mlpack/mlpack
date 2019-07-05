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
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

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
// BOOST_AUTO_TEST_CASE(NEATXORTest)
// {
//   arma::arma_rng::set_seed_random();
//   XORTask task;
//   NEAT<XORTask, HardSigmoidFunction, RankSelection> model(task, 2, 1, 100, 150, 10);
//   model.FinalFitness() = 3.8;

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = bestGenome.Fitness();
//   bestGenome.Print();
//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable.
//   BOOST_REQUIRE(finalFitness >= 3.5);
// }

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

// /**
//  * Test NEAT on the ContinuousMountainCar environment.
//  */
// BOOST_AUTO_TEST_CASE(NEATContinuousMountainCarTest)
// {
//   const ContinuousMountainCar env = ContinuousMountainCar();
//   ContinuousRLTask<ContinuousMountainCar> task(env);
//   NEAT<ContinuousRLTask<ContinuousMountainCar>, HardSigmoidFunction, RankSelection> model
//       (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, false);

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = bestGenome.Fitness();
//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

// /**
//  * Test NEAT on the MountainCar environment.
//  */
// BOOST_AUTO_TEST_CASE(NEATMountainCarTest)
// {
//   const MountainCar env = MountainCar();
//   DiscreteRLTask<MountainCar> task(env);
//   NEAT<DiscreteRLTask<MountainCar>, HardSigmoidFunction, RankSelection> model
//       (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = task.Evaluate(bestGenome);
//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

// /**
//  * Test NEAT on the Acrobot environment.
//  */
// BOOST_AUTO_TEST_CASE(NEATAcrobotTest)
// {
//   const Acrobot env = Acrobot();
//   DiscreteRLTask<Acrobot> task(env);
//   NEAT<DiscreteRLTask<Acrobot>, HardSigmoidFunction, RankSelection> model
//       (task, 2, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, true);

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = task.Evaluate(bestGenome);
//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

// /**
//  * Test NEAT on the CartPole environment.
//  */
// BOOST_AUTO_TEST_CASE(NEATCartPoleTest)
// {
//   const CartPole env = CartPole();
//   DiscreteRLTask<CartPole> task(env);
//   NEAT<DiscreteRLTask<CartPole>, HardSigmoidFunction, RankSelection> model
//        (task, 4, 1, 100, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, false);

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = task.Evaluate(bestGenome);
//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

/**
 * Test NEAT on the Double Pole Cart Balancing environment.
 */
// BOOST_AUTO_TEST_CASE(NEATDoublePoleCartTest)
// {
//   arma::vec poleLengths = {1, 0.1};
//   arma::vec poleMasses = {1, 0.1};
//   const MultiplePoleCart env = MultiplePoleCart(2, poleLengths, poleMasses);
//   DPVTask task(env);
//   NEAT<DPVTask, HardSigmoidFunction, RankSelection> model
//          (task, 6, 1, 150, 150, 10, 0.5, 0.8, 0.5, 0.8, 0.5, 0.2, 0.5, 0.3, 0.2, 0.5, false);

//   // Find the best genome and it's fitness.
//   Genome<HardSigmoidFunction> bestGenome = model.Train();
//   double finalFitness = bestGenome.Fitness();

//   std::cout << finalFitness << std::endl;

//   // Check if the final fitness is acceptable (Placeholder).
//   BOOST_REQUIRE(finalFitness >= 90);
// }

/**
 * Test NEAT on the Double Pole Cart Balancing environment without velocities.
 */
BOOST_AUTO_TEST_CASE(NEATDoublePoleCartNoVelocitiesTest)
{
  arma::arma_rng::set_seed_random();
  arma::vec poleLengths = {0.5, 0.05};
  arma::vec poleMasses = {0.1, 0.01};
  const MultiplePoleCart env = MultiplePoleCart(2, poleLengths, poleMasses, 9.8, 1, 0.01);
  DPNVTask task(env);
  NEAT<DPNVTask, LogisticFunction, RankSelection> model
        (task, 3, 1, 1000, 300, 10, 0, 1, 0.8, 1.8, 0.5, 0.01, 0.3, 0.2, 0.05, 0);

  // Find the best genome and it's fitness.
  Genome<LogisticFunction> bestGenome = model.Train();
  double finalFitness = bestGenome.Fitness();
  std::cout << finalFitness << std::endl;

  // Check if the final fitness is acceptable (Placeholder).
  BOOST_REQUIRE(finalFitness >= 90);
}

BOOST_AUTO_TEST_SUITE_END()
