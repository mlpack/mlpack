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
using namespace mlpack::neat;

BOOST_AUTO_TEST_SUITE(NEATTest)

/**
 * A Task class that wraps an RL environment.
 */
template<class EnvironmentType>
class RLTask
{
  RLTask(EnvironmentType& environment) : environment(environment)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    // Set the initial state.
    State state = environment.InitialSample();
    genome.Input() = state.Data();

    double loss = 0;
    while (!environment.IsTerminal())
    {
      // In this case we expect that genome.Output()
      // returns a vector of length 1, that does represent
      // the action to take at this state.
      // In case genome will return a vector of length
      // number of actions we could search for the maximum
      // value and return the index of that value in the vector.
      Action action = static_cast<Action>(genome.Output()[0]);

      // Use the current action to get the next state.
      loss += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      genome.Input() = state.Data();
    }

    return loss;
  }
  private:
  EnvironmentType environment;
}

/**
 * Test NEAT on the XOR Test.
 */
BOOST_AUTO_TEST_CASE(XORTest)
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
 * Test NEAT on the Double Pole Balancing environment with velocities.
 */
BOOST_AUTO_TEST_CASE(DoublePoleBalancingTest)
{
    
}

