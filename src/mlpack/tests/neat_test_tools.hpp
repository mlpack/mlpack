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
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>

using namespace mlpack;
using namespace mlpack::neat;
using namespace mlpack::ann;

// A class that defines the XOR task.
class XORTask
{
 public:
  double Evaluate(Genome<HardSigmoidFunction>& genome)
  {
    // // Create a random input of 0s and 1s.
    // arma::vec input = arma::randi<arma::vec>(2, arma::distr_param(0, 1));
    // arma::vec output = genome.Evaluate(input);

    // // The expected output of the XOR gate.
    // arma::vec answer = {(input[0] + input[1]) * (!input[0] + !input[1])};

    // // The fitness of the genome.
    // return 4 - std::pow(answer[0] - output[0], 2);

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

  double findError(arma::vec& input, Genome<HardSigmoidFunction>& genome)
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

// /**
//  * A Task class that wraps a continuous RL environment.
//  */
// template<class EnvironmentType>
// class ContinuousRLTask
// {
//   ContinuousRLTask(const EnvironmentType& env) : environment(env)
//   { /* Nothing to do here */ }

//   double Evaluate(Genome& genome)
//   {
//     // Set the initial state.
//     typename EnvironmentType::State state = environment.InitialSample();
//     genome.Input(state.Data());

//     double fitness = 0;
//     while (!environment.IsTerminal())
//     {
//       // Choose the action to perform.
//       typename EnvironmentType::Action action;
//       action.action[0] = genome.Output()[0];

//       // Use the current action to get the next state.
//       fitness += environment.Sample(state, action, state);

//       // Update the state of the genome for the next step.
//       genome.Input(state.Data());
//     }
//     return fitness;
//   }

//  private:
//   EnvironmentType environment;
// };

// /**
//  * A Task class that wraps a discrete RL environment.
//  */
// template<class EnvironmentType>
// class DiscreteRLTask
// {
//   DiscreteRLTask(const EnvironmentType& env) : environment(env)
//   { /* Nothing to do here */ }

//   double Evaluate(Genome& genome)
//   {
//     // Set the initial state.
//     typename EnvironmentType::State state = environment.InitialSample();
//     genome.Input(state.Data());

//     double fitness = 0;
//     while (!environment.IsTerminal())
//     {
//       // Choose the action to perform.
//       const int size = EnvironmentType::Action::size;
//       int actionInt = std::round(arma::clamp(genome.Output(), 0, size - 1)[0];
//       typename EnvironmentType::Action action = static_cast<typename
//           EnvironmentType::Action>(actionInt);

//       // Use the current action to get the next state.
//       fitness += environment.Sample(state, action, state);

//       // Update the state of the genome for the next step.
//       genome.Input(state.Data());
//     }
//     return fitness;
//   }

//  private:
//   EnvironmentType environment;
// };

// // A class that defines the Double Pole Balancing Task with velocities.
// class DPVTask
// {
//   DPVTask(const ContinuousMultiplePoleCart env)::environment(env)
//   { /* Nothing to do here */ }

//   double Evaluate(Genome& genome)
//   {
//     ContinuousMultiplePoleCart::State state = environment.InitialSample();
//     arma::mat inputMatrix = state.Data();
//     double angleThreshold = 12 * 2 * 3.1416 / 360;

//     // Input a vector scaled down features.
//     arma::vec input = {inputMatrix[0, 0] / 2.4, inputMatrix[1, 0] / 100,
//         inputMatrix[0, 1] / angleThreshold, inputMatrix[1, 1] / 100,
//         inputMatrix[0, 2] / angleThreshold, inputMatrix[1, 2] / 100};
//     genome.Input(input);

//     double fitnessDenom = 0;
//     int timeStep = 0;
//     while (!environment.IsTerminal())
//     {
//       timeStep++;

//       // Choose an action.
//       ContinuousMultiplePoleCart::Action action;
//       action.action[0] = genome.Output()[0];

//       // Use the current action to get the next state.
//       environment.Sample(state, action, state);

//       // Update the state of the genome for the next step.
//       inputMatrix = state.Data();
//       arma::vec input = {inputMatrix[0, 0] / 2.4, inputMatrix[1, 0] / 100,
//           inputMatrix[0, 1] / angleThreshold, inputMatrix[1, 1] / 100,
//           inputMatrix[0, 2] / angleThreshold, inputMatrix[1, 2] / 100};
//       genome.Input(input);

//       if (timeStep >= 1000)
//         continue;

//       if (timeStep >= 100)
//       {
//         int pow = timeStep - 100;
//         arma::vec temp = {inputMatrix[0, 0], inputMatrix[1, 0],
//             inputMatrix[0, 1], inputMatrix[1, 1]};
//         fitnessDenom += arma::accu(arma::pow(temp, pow));
//       }
//     }
//     return timeStep / 10000 + 0.675 / fitnessDenom;
//   }

//  private:
//   ContinuousMultiplePoleCart environment;
// };

// // A class that defines the Double Pole Balancing Task without velocities.
// class DPNVTask
// {
//   DPNVTask(const ContinuousMultiplePoleCart env)::environment(env)
//   { /* Nothing to do here */ }

//   double Evaluate(Genome& genome)
//   {
//     ContinuousMultiplePoleCart::State state = environment.InitialSample();
//     arma::mat inputMatrix = state.Data();
//     arma::vec input = inputMatrix.row(0);
//     genome.Input(input);

//     double fitnessDenom = 0;
//     int timeStep = 0;
//     while (!environment.IsTerminal())
//     {
//       timeStep++;

//       // Choose an action.
//       ContinuousMultiplePoleCart::Action action;
//       action.action[0] = genome.Output()[0];

//       // Use the current action to get the next state.
//       environment.Sample(state, action, state);

//       // Update the state of the genome for the next step.
//       inputMatrix = state.Data();
//       input = inputMatrix.row(0);

//       // Scale the input between -1 and 1.
//       input[0] /= 2.4;
//       input[1] /= 12 * 2 * 3.1416 / 360;
//       input[2] /= 12 * 2 * 3.1416 / 360;
//       genome.Input(input);

//       if (timeStep >= 1000)
//         continue;

//       if (timeStep >= 100)
//       {
//         int pow = timeStep - 100;
//         arma::vec temp = {inputMatrix[0, 0], inputMatrix[1, 0],
//             inputMatrix[0, 1], inputMatrix[1, 1]};
//         fitnessDenom += arma::accu(arma::pow(temp, pow));
//       }
//     }
//     return timeStep / 10000 + 0.675 / fitnessDenom;
//   }

//  private:
//   ContinuousMultiplePoleCart environment;
// };

#endif
