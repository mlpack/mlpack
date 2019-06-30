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
#include <mlpack/methods/neat/neat.hpp>

using namespace mlpack;
using namespace mlpack::neat;
using namespace mlpack::ann;
using namespace mlpack::rl;

// A class that defines the XOR task.
class XORTask
{
 public:
  static double Evaluate(Genome<HardSigmoidFunction>& genome)
  {
    arma::vec input1 = {0, 0};
    arma::vec input2 = {0, 1};
    arma::vec input3 = {1, 0};
    arma::vec input4 = {1, 1};
    double error = 0;
    error += findError(input1, genome); 
    error += findError(input2, genome); 
    error += findError(input3, genome); 
    error += findError(input4, genome);
    return 4 - error;
  }

  static double findError(arma::vec& input, Genome<HardSigmoidFunction>& genome)
  {
    arma::vec output = genome.Evaluate(input);
    arma::vec answer = {(input[0] + input[1]) * (!input[0] + !input[1])};
    return std::pow(answer[0] - output[0], 2);
  }

  void findOutput(arma::vec& input, Genome<HardSigmoidFunction>& genome)
  {
    arma::vec output = genome.Evaluate(input);
    input.print();
    std::cout << "=>" << std::endl;
    output.print();
  }
};

/**
 * A Task class that wraps a continuous RL environment.
 */
template<class EnvironmentType>
class ContinuousRLTask
{
 public:
  ContinuousRLTask(const EnvironmentType& env) : environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome<HardSigmoidFunction>& genome)
  {
    // Set the initial state.
    typename EnvironmentType::State state = environment.InitialSample();
    arma::vec output = genome.Evaluate(state.Data());

    double fitness = 0;
    size_t k = 0;
    while (!environment.IsTerminal(state) && k < 1000)
    {
      k++;
      // Choose the action to perform.
      typename EnvironmentType::Action action;
      action.action[0] = output[0];

      // Use the current action to get the next state.
      fitness += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      output = genome.Evaluate(state.Data());
    }
    return 10 + fitness;
  }

 private:
  EnvironmentType environment;
};

/**
 * A Task class that wraps a discrete RL environment.
 */
template<class EnvironmentType>
class DiscreteRLTask
{
 public:
  DiscreteRLTask(const EnvironmentType& env) : environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome<HardSigmoidFunction>& genome)
  {
    // Set the initial state.
    typename EnvironmentType::State state = environment.InitialSample();
    arma::vec output = genome.Evaluate(state.Data());

    double fitness = 0;
    while (!environment.IsTerminal(state))
    {
      // Choose the action to perform.
      const int size = EnvironmentType::Action::size;
      output = arma::clamp(output, 0, size - 1);
      int actionInt = std::round(output[0]);
      typename EnvironmentType::Action action = static_cast<typename
          EnvironmentType::Action>(actionInt);

      // Use the current action to get the next state.
      fitness += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      output = genome.Evaluate(state.Data());
    }
    return fitness;
  }

 private:
  EnvironmentType environment;
};

// A class that defines the Double Pole Balancing Task with velocities.
class DPVTask
{
 public:
  DPVTask(const MultiplePoleCart env) : environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome<HardSigmoidFunction>& genome)
  {
    MultiplePoleCart::State state = environment.InitialSample();
    arma::mat inputMatrix = state.Data();
    double angleThreshold = 36 * 2 * 3.1416 / 360;

    // Input a vector scaled down features.
    arma::vec input = {inputMatrix(0,0) / 2.4, inputMatrix(1, 0),
          inputMatrix(0, 1) / angleThreshold, inputMatrix(1, 1),
          inputMatrix(0, 2) / angleThreshold, inputMatrix(1, 2)};
    arma::vec output = genome.Evaluate(input);

    double fitnessDenom = 0;
    int timeStep = 0;
    while (!environment.IsTerminal(state))
    {
      timeStep++;

      // Choose an action.
      const int size = MultiplePoleCart::Action::size;
      output = arma::clamp(output, 0, size - 1);
      int actionInt = std::round(output[0]);
      MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>(actionInt);

      // Use the current action to get the next state.
      environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      inputMatrix = state.Data();
      arma::vec input = {inputMatrix(0,0) / 2.4, inputMatrix(1, 0),
          inputMatrix(0, 1) / angleThreshold, inputMatrix(1, 1),
          inputMatrix(0, 2) / angleThreshold, inputMatrix(1, 2)};
      output = genome.Evaluate(input);

      if (timeStep > 10000)
        break;
    }
    //return timeStep / 10000 + 0.675 / fitnessDenom;
    std::cout << timeStep << std::endl;
    return timeStep;
  }

 private:
  MultiplePoleCart environment;
};

// A class that defines the Double Pole Balancing Task without velocities.
class DPNVTask
{
 public:
  DPNVTask(const MultiplePoleCart& env) : environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome<HardSigmoidFunction>& genome)
  {
    double fitness = 0;
    for (size_t j = 0; j < 20; j++)
    {
      MultiplePoleCart::State state = environment.InitialSample();
      arma::mat inputMatrix = state.Data();
      arma::vec input = {inputMatrix(0, 0), inputMatrix(0, 1), inputMatrix(0, 2)};
      arma::vec output = genome.Evaluate(input);
      std::deque<double> wiggleBuffer1;
      std::deque<double> wiggleBuffer2;
      int timeStep = 0;
      while (!environment.IsTerminal(state))
      {
        timeStep++;

        const int size = MultiplePoleCart::Action::size;
        output = arma::clamp(output, 0, size - 1);
        int actionInt = std::round(output[0]);
        MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>(actionInt);

        // Use the current action to get the next state.
        environment.Sample(state, action, state);

        // Update the state of the genome for the next step.
        inputMatrix = state.Data();
        input = {inputMatrix(0, 0), inputMatrix(0, 1), inputMatrix(0, 2)};

        // Scale the input between -1 and 1.
        input[0] /= 2.4;
        input[1] /= 36 * 2 * 3.1416 / 360;
        input[2] /= 36 * 2 * 3.1416 / 360;
        
        wiggleBuffer1.push_back(std::abs(inputMatrix(0,0)) + std::abs(inputMatrix(0,1)) +
            std::abs(inputMatrix(1,0)) + std::abs(inputMatrix(1,1)));
        
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
    }

    std::cout <<" Fitness: " << fitness/20 << std::endl;
    return fitness / 20;
  }

 private:
  MultiplePoleCart environment;
};

#endif
