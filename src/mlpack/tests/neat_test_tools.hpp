/**
 * @file neat_test_tools.hpp
 * @author Rahul Ganesh Prabhu
 *
 * This file includes some useful classes for neat tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_NEAT_TEST_TOOLS_HPP
#define MLPACK_TESTS_NEAT_TEST_TOOLS_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp>
#include <mlpack/methods/neat/neat.hpp>

using namespace mlpack;
using namespace mlpack::neat;
using namespace mlpack::ann;
using namespace mlpack::rl;

/**
 * A class that defines the XOR task. 
 */
class XORTask
{
 public:
  double Evaluate(Genome<>& genome)
  {
    arma::mat input = {{0, 0, 1, 1},
                       {0, 1, 0, 1}};

    double error = 0;
    for (size_t i = 0; i < input.n_cols; i++)
    {
      arma::vec inputVec = input.col(i);
      arma::vec output = genome.Evaluate(inputVec);
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

  double Evaluate(Genome<>& genome)
  {
    double fitness = 0;

    // The starting point is always the same.
    double oneDegrees = 1 * M_PI / 180;
    arma::vec data(6, arma::fill::zeros);
    data[2] = oneDegrees;
    MultiplePoleCart::State state(data);

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

#endif
