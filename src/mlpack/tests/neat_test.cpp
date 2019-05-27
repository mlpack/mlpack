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

#include <mlpack/methods/neat/neat.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/continuous_mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/environment/multiple_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/environment/acrobot.hpp>
#include <mlpack/methods/reinforcement_learning/environment/pendulum.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;
using namespace mlpack::neat;

BOOST_AUTO_TEST_SUITE(NEATTest)

/**
 * A Task class that wraps a continuous RL environment.
 */
template<class EnvironmentType>
class ContinuousRLTask
{
  ContinuousRLTask(EnvironmentType& environment) : environment(environment)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    // Set the initial state.
    EnvironmentType::State state = environment.InitialSample();
    genome.Input(state.Data());

    double loss = 0;
    while (!environment.IsTerminal())
    {

      EnvironmentType::Action action;
      action.action[0] = genome.Output()[0];

      // Use the current action to get the next state.
      loss += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      genome.Input(state.Data());
    }

    return loss;
  }
 private:
  EnvironmentType environment;
}

/**
 * A Task class that wraps a discrete RL environment.
 */
template<class EnvironmentType>
class DiscreteRLTask
{
  DiscreteRLTask(EnvironmentType& environment) : environment(environment)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    // Set the initial state.
    EnvironmentType::State state = environment.InitialSample();
    genome.Input(state.Data());

    double loss = 0;
    while (!environment.IsTerminal())
    {
      const size = EnvironmentType::Action::size;
      EnvironmentType::Action action = static_cast<EnvironmentType::Action>(
        std::round(arma::clamp(genome.Output(), 0, size - 1))[0]);

      // Use the current action to get the next state.
      loss -= environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      genome.Input(state.Data());
    }

    return loss;
  }
 private:
  EnvironmentType environment;
}

/**
 * Test NEAT on the XOR Test.
 */
BOOST_AUTO_TEST_CASE(NEATXORTest)
{
  // A class that defines the XOR task.
  class Task
  {
    double Evaluate(Genome& genome)
    {
      // Create a random input of 0s and 1s.
      arma::vec input = arma::randi<arma::vec>(1);
      genome.Input(input);
      arma::vec output = genome.Output();
      
      // The expected output of the XOR gate.
      arma::vec answer = ((input[0] + input[1])*(!input[0] + input[1])); 
      
      // The loss of the genome. 
      double loss = std::pow(answer[0] - output[0], 2);
      return loss;
    }
  }

  Task task();
  NEAT<Task> model(task, 2, 1);
  // Find the best genome.
  Genome bestGenome = model.Train();
  
  double finalLoss = task.Evaluate(bestGenome)

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the Pendulum environment.
 */
BOOST_AUTO_TEST_CASE(NEATPendulumTest)
{
  const Pendulum env = Pendulum();
  ContinuousRLTask<Pendulum> task(env);
  NEAT<ContinuousRLTask<Pendulum>> model(task, 2, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the ContinuousMountainCar environment.
 */
BOOST_AUTO_TEST_CASE(NEATContinuousMountainCarTest)
{
  const ContinuousMountainCar task = ContinuousMountainCar();
  ContinuousRLTask<ContinuousMountainCar> task(env);
  NEAT<ContinuousRLTask<ContinuousMountainCar>> model(task, 2, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the MountainCar environment.
 */
BOOST_AUTO_TEST_CASE(NEATMountainCarTest)
{
  const MountainCar env = MountainCar();
  DiscreteRLTask<MountainCar> task(env);
  NEAT<DiscreteRLTask<MountainCar>> model(task, 2, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the Acrobot environment.
 */
BOOST_AUTO_TEST_CASE(NEATAcrobotTest)
{
  const Acrobot env = Acrobot();
  DiscreteRLTask<Acrobot> task(env);
  NEAT<DiscreteRLTask<Acrobot>> model(task, 2, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the CartPole environment.
 */
BOOST_AUTO_TEST_CASE(NEATCartPoleTest)
{
  const CartPole env = CartPole();
  DiscreteRLTask<CartPole> task(env);
  NEAT<DiscreteRLTask<CartPole>> model(task, 2, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the Double Pole Cart Balancing environment.
 */
BOOST_AUTO_TEST_CASE(NEATDoublePoleCartTest)
{
  class Task
  {
    Task(const MultiplePoleCart env)::environment(env)
    { /* Nothing to do here */ }

    double Evaluate(Genome& genome)
    {
      EnvironmentType::State state = environment.InitialSample();
      arma::mat input = state.Data();
      genome.Input(input.as_col());

      double loss = 0;
      while (!environment.IsTerminal())
      {
        MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>(
        std::round(arma::clamp(genome.Output(), 0, 3))[0]);

        // Use the current action to get the next state.
        loss += environment.Sample(state, action, state);

        // Update the state of the genome for the next step.
        input = state.Data();
        genome.Input(input.as_col());
      }

      return loss;
    }

   private:
    MultiplePoleCart environment;
  }

  arma::vec poleLengths = {1, 0.5};
  arma::vec poleMasses = {1, 1};
  const MultiplePoleCart env = MultiplePoleCart(2, poleLengths, poleMasses);
  Task task(env);
  NEAT<Task> model(task, 6, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

/**
 * Test NEAT on the Double Pole Cart Balancing environment without velocities.
 */
BOOST_AUTO_TEST_CASE(NEATDoublePoleCartNoVelocitiesTest)
{
  class Task
  {
    Task(const MultiplePoleCart env)::environment(env)
    { /* Nothing to do here */ }

    double Evaluate(Genome& genome)
    {
      EnvironmentType::State state = environment.InitialSample();
      arma::mat input = state.Data();
      // Remove the velocity parameter.
      genome.Input(input.as_col().shed_row(1));

      double loss = 0;
      while (!environment.IsTerminal())
      {
        // Choose an action.
        MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>(
          std::round(arma::clamp(genome.Output(), 0, 3))[0]);

        // Use the current action to get the next state.
        loss += environment.Sample(state, action, state);

        // Update the state of the genome for the next step.
        input = state.Data();
        genome.Input(input.as_col().shed_row(1));
      }
      return loss;
    }

   private:
    MultiplePoleCart environment;
  }

  arma::vec poleLengths = {1, 0.5};
  arma::vec poleMasses = {1, 1};
  const MultiplePoleCart env = MultiplePoleCart(2, poleLengths, poleMasses);
  Task task(env);
  NEAT<Task> model(task, 5, 1);
  Genome bestGenome = model.Train();
  double finalLoss = task.Evaluate(bestGenome);

  // Check if the final loss is acceptable.
  BOOST_REQUIRE(finalLoss <= 0.1);
}

BOOST_AUTO_TEST_SUITE_END()