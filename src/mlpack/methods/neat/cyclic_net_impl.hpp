/**
 * @file cyclic_net_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the CyclicNet class, which represents a cyclic neural
 * network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_CYCLIC_NET_IMPL_HPP
#define MLPACK_METHODS_NEAT_CYCLIC_NET_IMPL_HPP

// In case it hasn't been included yet.
#include "cyclic_net.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

template <class ActivationFunction>
CyclicNet<ActivationFunction>::CyclicNet(std::map<size_t, std::map<size_t,
                                            ConnectionGene>>& directedGraph,
                                         ActivationFunction& actFn,
                                         const size_t nodeCount,
                                         const size_t inputNodeCount,
                                         const size_t outputNodeCount,
                                         const size_t timeStepsPerActivation,
                                         const double bias):
    directedGraph(directedGraph),
    actFn(actFn),
    nodeCount(nodeCount),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    timeStepsPerActivation(timeStepsPerActivation),
    bias(bias)
{
  for (size_t i = 0; i < nodeCount; i++)
    outputNodeValues.push_back(0);
}

template <class ActivationFunction>
arma::vec CyclicNet<ActivationFunction>::Evaluate(arma::vec input)
{
  std::vector<double> inputNodeValues(nodeCount, 0);

  // Load the bias.
  outputNodeValues[0] = bias;

  // Load the input.
  for (size_t i = 1; i <= inputNodeCount; i++)
    outputNodeValues[i] = input[i - 1];

  for (size_t i = 0; i < timeStepsPerActivation; i++)
  {
    for (size_t j = 0; j < nodeCount; j++)
    {
      for (auto const &x : directedGraph[j])
      {
        double weight = x.second.getWeight();
        inputNodeValues[x.first] += nodeValues[j] * weight;
      }
    }

    for (size_t i = inputNodeCount; i < nodeCount; i++)
    {
      nodeValues[i] = actFn.Fn(inputNodeValues[i]);
    }
  }

  arma::vec output(outputNodeCount);
  for (size_t i = 0; i < output.n_elem; i++)
    output[i] = putputNodeValues[i + inputNodeCount + 1];

  return output;
}

} // namespace neat
} // namespace mlpack

#endif
