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
#include <mlpack/methods/reinforcement_learning/environment/continuous_multiple_pole_cart.hpp>
#include <mlpack/methods/reinforcement_learning/environment/multiple_pole_cart.hpp>
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
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
  double Evaluate(Genome<HardSigmoidFunction>& genome)
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
  DPNVTask(const MultiplePoleCart& env) : environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome<>& genome)
  {
    double fitness = 0;

    // The starting point is always the same.
    double oneDegrees = 1 * M_PI / 180;
    arma::mat data(2, 3, arma::fill::zeros);
    data(0, 1) = oneDegrees;
    MultiplePoleCart::State state(data);
    arma::mat inputMatrix = state.Data();

    // Create a normalized vector input.
    arma::vec input = {inputMatrix(0, 0) / 2.4, inputMatrix(0, 1) / 0.62832,
        inputMatrix(0, 2) / 0.62832};
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
      const int size = MultiplePoleCart::Action::size;
      output = arma::clamp(output, 0, size - 1);
      int actionInt = std::round(output[0]);
      MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>
          (actionInt);

      // Use the current action to get the next state.
      environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      inputMatrix = state.Data();
      input = {inputMatrix(0, 0), inputMatrix(0, 1), inputMatrix(0, 2)};

      // Scale the input between -1 and 1.
      input[0] /= 2.4;
      input[1] /= 0.62832;
      input[2] /= 0.62832;

      wiggleBuffer1.push_back(std::abs(inputMatrix(0, 0)) +
          std::abs(inputMatrix(0, 1)) + std::abs(inputMatrix(1, 0)) +
          std::abs(inputMatrix(1, 1)));

      if (wiggleBuffer1.size() == 100)
      {
        wiggleBuffer2.push_back(wiggleBuffer1.front());
        wiggleBuffer1.pop_front();
      }
      if (wiggleBuffer2.size() == 100)
        wiggleBuffer2.pop_front();

      output = genome.Evaluate(input);

      if (timeStep > 499)
      {
        double sum = 0;
        for (auto it = wiggleBuffer2.begin(); it != wiggleBuffer2.end(); it++)
          sum += *it;
        if (sum > 30)
          break;
      }
    }

    if(timeStep > 499 && timeStep < 600)
    {
      double sum = 0;
      for (auto it = wiggleBuffer1.begin(); it != wiggleBuffer1.end(); it++)
        sum += *it;
      fitness += timeStep + (10.0 / std::max(1.0, sum));
    }
    else if(timeStep > 599)
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
  MultiplePoleCart environment;
};

#endif
