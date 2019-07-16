/**
 * @file acyclic_net_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the AcyclicNet class, which representa an acyclic neural
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
void AcyclicNet<ActivationFunction>::Evaluate(arma::vec& input,
                                              arma::vec& output,
                                              std::map<size_t, std::map<size_t,
                                                ConnectionGene>>& directedGraph,
                                              std::vector<size_t>& nodeDepths)
{
  arma::vec nodeValues(nodeCount, arma::fill::zeros);

  // Populate the layers.
  for (size_t i = 0; i < nodeCount; i++)
  {
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
      if (nodeID == 0)
      {
        for (auto const& x : directedGraph[nodeID])
        {
          if (x.second.Enabled())
            nodeValues[x.first] += bias * x.second.Weight();
        }
      }
      else if (nodeID <= inputNodeCount)
      {
        for (auto const& x : directedGraph[nodeID])
        {
          if (x.second.Enabled())
            nodeValues[x.first] += nodeValues[nodeID] * x.second.Weight();
        }
      }
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
