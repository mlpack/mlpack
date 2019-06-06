/**
 * @file acyclic_net_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the acyclic net class.
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

template <class ActivationFunction>
AcyclicNet<ActivationFunction>::AcyclicNet(std::vector<size_t>& nodeGeneList,
                                           std::map<size_t, std::map<size_t,
                                              ConnectionGene>>& directedGraph,
                                           ActivationFunction& actFn,
                                           const size_t inputNodeCount,
                                           const size_t outputNodeCount,
                                           const double bias):
    nodeGeneList(nodeGeneList),
    directedGraph(directedGraph),
    actFn(actFn),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    bias(bias)
{
  // Find the depth of the nodes.
  for (size_t i = 0; i <= inputNodeCount; i++)
  {
    TraverseNode(i, 0);
  }

  // Populate the layers.
  for (auto const& x : nodeDepths)
  {
    while (layers.size() < x.second + 1)
      layers.emplace_back(std::vector<size_t>());
    layers[x.second].push_back(x.first);
  }
}

// Recursively traverse neighbours and assign depths.
template <class ActivationFunction>
void AcyclicNet<ActivationFunction>::TraverseNode(size_t nodeID, size_t depth)
{
  if (nodeDepths.find(nodeID) != nodeDepths.end())
  {
    // Check if it has been traversed by a longer path.
    if (nodeDepths[nodeID] >= depth)
      return;
    else
      nodeDepths[nodeID] = depth;
  }
  else
  {
    nodeDepths.insert(nodeID, depth);
  }

  for (auto const& x : directedGraph)
    TraverseNode(x.first, depth + 1);
}

// Evaluate a given input.
template <class ActivationFunction>
arma::vec AcyclicNet<ActivationFunction>::Evaluate(arma::vec input)
{
  std::map<size_t, double> nodeValues;
  // Add all the nodes to the map.
  for (size_t i = 0; i < nodeGeneList.size(); i++)
    if (nodeGeneList[i] <= inputNodeCount && i != 0)
      nodeValues[nodeGeneList[i]] = input[i-1];
    else
      nodeValues[nodeGeneList[i]] = 0;

  // Activate the layers one by one.
  for (size_t i = 0; i < layers.size(); i++)
  {
    for (size_t j = 0; j < layers[i].size(); j++)
    {
      int nodeID = layers[i][j];
      if (nodeID == 0)
      {
        for (auto const& x : directedGraph[nodeID])
          nodeValues[x.first] += bias * x.second.getWeight();
      }
      else
      {
        double result = actFn.Fn(nodeValues[nodeID]);
        for (auto const& x : directedGraph)
          nodeValues[x.first] += result * x.second.getWeight();
      }
    }
  }

  // Find the output.
  arma::vec output(outputNodeCount);
  for (size_t i = 0; i < output.n_elem; i++)
    output[i] = nodeValues[i + inputNodeCount + 1];

  return output;
}

} // namespace neat
} // namespace mlpack

#endif
