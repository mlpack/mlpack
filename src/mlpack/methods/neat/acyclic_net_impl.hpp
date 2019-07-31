/**
 * @file acyclic_net_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the AcyclicNet class, which represents an acyclic neural
 * network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_ACYCLIC_NET_IMPL_HPP
#define MLPACK_METHODS_NEAT_ACYCLIC_NET_IMPL_HPP

// In case it hasn't been included yet.
#include "acyclic_net.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

// Creates an AcyclicNet object.
template <class ActivationFunction>
AcyclicNet<ActivationFunction>::AcyclicNet(const size_t nodeCount,
                                           const size_t inputNodeCount,
                                           const size_t outputNodeCount,
                                           const double bias):
    nodeCount(nodeCount),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    bias(bias)
{ /* Nothing to do here */ }

// Evaluate a given input.
template <class ActivationFunction>
void AcyclicNet<ActivationFunction>::Evaluate(const arma::vec& input,
                                              arma::vec& output,
                                              std::map<size_t, std::map<size_t,
                                                ConnectionGene>>& directedGraph,
                                              std::vector<size_t>& nodeDepths)
{
  arma::vec nodeValues(nodeCount, arma::fill::zeros);

  // Populate the layers.
  for (size_t i = 0; i < nodeCount; i++)
  {
    layers.reserve(nodeDepths[i] + 1);
    while (layers.size() < nodeDepths[i] + 1)
      layers.emplace_back(std::vector<size_t>());
    layers[nodeDepths[i]].push_back(i);
  }

  nodeValues[0] = bias;

  // Add all the nodes to the map.
  for (size_t i = 1; i <= inputNodeCount; i++)
    nodeValues[i] = input[i - 1];

  // Activate the layers one by one.
  for (size_t i = 0; i < layers.size(); i++)
  {
    for (size_t j = 0; j < layers[i].size(); j++)
    {
      size_t nodeID = layers[i][j];
      // If this is a bias node, we need to add bias to it's connections.
      if (nodeID == 0)
      {
        for (auto const& x : directedGraph[nodeID])
        {
          if (x.second.Enabled())
            nodeValues[x.first] += bias * x.second.Weight();
        }
      }
      // If it is an input node, we need not apply the activation function.
      else if (nodeID <= inputNodeCount)
      {
        for (auto const& x : directedGraph[nodeID])
        {
          if (x.second.Enabled())
            nodeValues[x.first] += nodeValues[nodeID] * x.second.Weight();
        }
      }
      // In all other cases, we can proceed normally.
      else
      {
        double result = ActivationFunction::Fn(nodeValues[nodeID]);
        for (auto const& x : directedGraph[nodeID])
        {
          if (x.second.Enabled())
            nodeValues[x.first] += result * x.second.Weight();
        }
      }
    }
  }

  for (size_t i = 0; i < output.n_elem; i++)
    output[i] = ActivationFunction::Fn(nodeValues[i + inputNodeCount + 1]);
}

} // namespace neat
} // namespace mlpack

#endif
