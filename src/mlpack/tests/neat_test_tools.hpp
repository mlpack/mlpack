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

using namespace mlpack;
using namespace mlpack::neat;

// A class that defines the XOR task.
class XORTask
{
  double Evaluate(const Genome& genome)
  {
    // Create a random input of 0s and 1s.
    arma::vec input = arma::randi<arma::vec>(2, distr_param(0, 1));
    genome.Input(input);
    arma::vec output = genome.Output();
    
    // The expected output of the XOR gate.
    arma::vec answer = {(input[0] + input[1]) * (!input[0] + input[1])}; 
    
    // The fitness of the genome. 
    return 4 - std::pow(answer[0] - output[0], 2);
  }
};

/**
 * A Task class that wraps a continuous RL environment.
 */
template<class EnvironmentType>
class ContinuousRLTask
{
  ContinuousRLTask(const EnvironmentType& environment) : environment(environment)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    // Set the initial state.
    typename EnvironmentType::State state = environment.InitialSample();
    genome.Input(state.Data());

    double fitness = 0;
    while (!environment.IsTerminal())
    {

      typename EnvironmentType::Action action;
      action.action[0] = genome.Output()[0];

      // Use the current action to get the next state.
      fitness += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      genome.Input(state.Data());
    }
    return fitness;
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
  DiscreteRLTask(const EnvironmentType& environment) : environment(environment)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    // Set the initial state.
    typename EnvironmentType::State state = environment.InitialSample();
    genome.Input(state.Data());

    double fitness = 0;
    while (!environment.IsTerminal())
    {
      const int size = EnvironmentType::Action::size;
      typename EnvironmentType::Action action = static_cast<typename EnvironmentType::Action>(
        std::round(arma::clamp(genome.Output(), 0, size - 1))[0]);

      // Use the current action to get the next state.
      fitness += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      genome.Input(state.Data());
    }
    return fitness;
  }

 private:
  EnvironmentType environment;
};

// A class that defines the Double Pole Balancing Task with velocities.
class DPVTask
{
  DPVTask(const MultiplePoleCart env)::environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    MultiplePoleCart::State state = environment.InitialSample();
    arma::mat inputMatrix = state.Data();
    arma::vec input = {inputMatrix[0, 0], inputMatrix[1, 0], inputMatrix[0, 1], 
        inputMatrix[1, 1], inputMatrix[0, 2]}, inputMatrix[1, 2]};
    genome.Input(input);

    double fitness = 0;
    while (!environment.IsTerminal())
    {
      timeStep++;
      MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>(
      std::round(arma::clamp(genome.Output(), 0, 3))[0]);

      // Use the current action to get the next state.
      fitness += environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      inputMatrix = state.Data();
      input = {inputMatrix[0, 0], inputMatrix[0, 1], inputMatrix[1, 0], 
          inputMatrix[1, 1], inputMatrix[2, 0]}, inputMatrix[2, 1]};
      genome.Input(input);
    }
    return fitness;
  }

 private:
  MultiplePoleCart environment;
};

// A class that defines the Double Pole Balancing Task without velocities.
class DPNVTask
{
  DPNVTask(const MultiplePoleCart env)::environment(env)
  { /* Nothing to do here */ }

  double Evaluate(Genome& genome)
  {
    MultiplePoleCart::State state = environment.InitialSample();
    arma::mat inputMatrix = state.Data();
    arma::vec input = inputMatrix.row(0);
    genome.Input(input);

    double fitnessDenom = 0;
    int timeStep = 0;
    while (!environment.IsTerminal())
    {
      timeStep++;
      // Choose an action.
      MultiplePoleCart::Action action = static_cast<MultiplePoleCart::Action>(
        std::round(arma::clamp(genome.Output(), 0, 3))[0]);

      // Use the current action to get the next state.
      environment.Sample(state, action, state);

      // Update the state of the genome for the next step.
      inputMatrix = state.Data();
      input = inputMatrix.row(0);

      // Scale the input between -1 and 1.
      input[0] /= 2.4;
      input[1] /= 12 * 2 * 3.1416 / 360;
      input[2] /= 12 * 2 * 3.1416 / 360;
      genome.Input(input);

      if(timeStep >= 100)
      {
        int pow = timeStep - 100;
        arma::vec temp = {inputMatrix[0, 0], inputMatrix[1, 0], inputMatrix[0, 1], 
            inputMatrix[1, 1]};
        fitnessDenom += arma::sum(arma::pow(temp, pow));
      } 
      if(timeStep >= 1000)
        break;
    }
    return timeStep / 10000 + 0.675 / fitnessDenom;
  }

 private:
  MultiplePoleCart environment;
};

#endif
