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
#include <mlpack/methods/neat/selection_strategies/rank_selection.hpp>
#include <mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::rl;
using namespace mlpack::ann;
using namespace mlpack::neat;

BOOST_AUTO_TEST_SUITE(NEATTest)

/**
 * A class that defines the XOR task.
 *
 * The expected output follows this scheme:
 * |  Input  |  Output  |
 * | {0, 0}  |    0     |
 * | {0, 1}  |    1     |
 * | {1, 0}  |    1     |
 * | {1, 1}  |    0     |
 *
 */
class XORTask
{
 public:
  template<typename GenomeType>
  double Evaluate(GenomeType& genome)
  {
    arma::mat input = {{0, 0, 1, 1},
                       {0, 1, 0, 1}};

    double error = 0;
    for (size_t i = 0; i < input.n_cols; i++)
    {
      arma::vec output = genome.Evaluate(input.col(i));
      arma::vec answer = {(input(0, i) + input(1, i)) * (!input(0, i) +
          !input(1, i))};
      error += std::pow(answer[0] - output[0], 2);
    }
    return 4 - error;
  }
};

/**
 * A class that defines the non-Markovian double pole balancing task. The
 * observation space is comprised of the cart position and the angles of the poles.
 */
class DPNVTask
{
 public:
  DPNVTask(const DoublePoleCart& env) : environment(env)
  { /* Nothing to do here */ }

  template<typename GenomeType>
  double Evaluate(GenomeType& genome)
  {
    double fitness = 0;

    // The starting point is always the same.
    double oneDegrees = 1 * M_PI / 180;
    arma::vec data(6, arma::fill::zeros);
    data[2] = oneDegrees;
    DoublePoleCart::State state(data);

    // Create a normalized vector input.
    arma::vec input = {state.Position() / 2.4, state.Angle(1) / 0.62832,
        state.Angle(2) / 0.62832};
    arma::vec output = genome.Evaluate(input);

    // Helper variables to evaluate fitness.
    std::deque<double> wiggleBuffer1;
    std::deque<double> wiggleBuffer2;
    int timeStep = 0;

    // Main loop.
    while (!environment.IsTerminal(state))
    {
      // Update number of timesteps.
      timeStep++;

      // Find the action to apply.
      const int size = DoublePoleCart::Action::size;
      output = arma::clamp(output, 0, size - 1);
      int actionInt = std::round(output[0]);
      DoublePoleCart::Action action = static_cast<DoublePoleCart::Action>
          (actionInt);

      // Use the current action to get the next state.
      environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      input = {state.Position() / 2.4, state.Angle(1) / 0.62832,
          state.Angle(2) / 0.62832};

      wiggleBuffer1.push_back(std::abs(state.Position()) +
          std::abs(state.Velocity()) + std::abs(state.Angle(1)) +
          std::abs(state.AngularVelocity(1)));

      if (wiggleBuffer1.size() == 100)
      {
        wiggleBuffer2.push_back(wiggleBuffer1.front());
        wiggleBuffer1.pop_front();
      }
      if (wiggleBuffer2.size() == 100)
        wiggleBuffer2.pop_front();

      output = genome.Evaluate(input);

      if (timeStep > 10000)
        break;
      else if (timeStep > 499)
      {
        double sum = 0;
        for (auto it = wiggleBuffer2.begin(); it != wiggleBuffer2.end(); it++)
          sum += *it;
        // This means the agent is just wiggling, let's stop this agent.
        if (sum > 30)
          break;
      }
    }

    if (timeStep > 499 && timeStep < 600)
    {
      double sum = 0;
      for (auto it = wiggleBuffer1.begin(); it != wiggleBuffer1.end(); it++)
        sum += *it;
      fitness += timeStep + (10.0 / std::max(1.0, sum));
    }
    else if (timeStep > 599)
    {
      double sum = 0;
      for (auto it = wiggleBuffer2.begin(); it != wiggleBuffer2.end(); it++)
        sum += *it;
      fitness += timeStep + (10.0 / std::max(1.0, sum));
    }
    else
      fitness += timeStep;

    return fitness;
  }

 private:
  DoublePoleCart environment;
};

/**
 * Test NEAT and phased searching on the XOR Test. XOR normally performs better
 * in cyclic cases, this is meant to be a test of the acyclic network.
 */
BOOST_AUTO_TEST_CASE(NEATXORTest)
{
  XORTask task;
  NEAT<XORTask> model(task, 2, 1, 100, 500, 10);
  model.FinalFitness() = 3.7;
  model.IsAcyclic() = true;
  model.ComplexityThreshold() = 6;

  // Find the best genome and it's fitness.
  Genome<> bestGenome;
  model.Train(bestGenome);
  double finalFitness = bestGenome.Fitness();
  double meanComplexity = model.MeanComplexity();

  // Check if the final fitness is acceptable.
  BOOST_REQUIRE(finalFitness >= 3.7);

  // Check if the complexity is not too far in excess of the complexity
  // ceiling.
  BOOST_REQUIRE(meanComplexity < (double)model.CurrentComplexityCeiling() + 1);
}

/**
 * Test NEAT on the Double Pole Cart Balancing environment without velocities.
 */
BOOST_AUTO_TEST_CASE(NEATDoublePoleCartNoVelocitiesTest)
{
  DoublePoleCart env = DoublePoleCart();
  DPNVTask task(env);
  NEAT<DPNVTask> model(task, 3, 1, 1000, 200, 50, 0, 1, 0.8, 1.8, 0.5, 0.01,
      0.3, 0.2, 0.05, 0);
  model.FinalFitness() = 10000;

  // Find the best genome and it's fitness.
  Genome<> bestGenome;
  model.Train(bestGenome);
  double finalFitness = bestGenome.Fitness();
  Log::Debug << "The final fitness is " << finalFitness << std::endl;

  // Check if the final fitness is acceptable.
  BOOST_REQUIRE(finalFitness >= 10000);
}

BOOST_AUTO_TEST_SUITE_END()
