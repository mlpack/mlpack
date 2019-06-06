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
AcyclicNet<ActivationFunction>::AcyclicNet(std::vector<size_t>& NodeGeneList,
                                           std::map<size_t, std::map<size_t,
                                              ConnectionGene>>& DirectedGraph,
                                           ActivationFunction& actFn,
                                           const size_t inputNodeCount,
                                           const size_t outputNodeCount,
                                           const double bias):
    NodeGeneList(NodeGeneList),
    DirectedGraph(DirectedGraph),
    actFn(actFn),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    bias(bias)
{
  // Find the depth of the nodes
  for (size_t i = 0; i <= inputNodeCount; i++)
  {
    TraverseNode(i, 0);
  }

  // Populate the layers
  for (auto const& x : nodeDepths)
  {
    while (layers.size() < x.second + 1)
      layers.emplace_back(std::vector<size_t>());
    layers[it->second].push_back(x.first);
  }
}

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

  for (auto const& x : DirectedGraph)
    TraverseNode(x.first, depth + 1);
}

template <class ActivationFunction>
arma::vec AcyclicNet<ActivationFunction>::Evaluate(arma::vec input)
{
  std::map<size_t, double> NodeValues;
  for (size_t i = 0; i < NodeGeneList.size(); i++)
    if (NodeGeneList[i] <= inputNodeCount)
      NodeValues.insert(NodeGeneList[i], input[NodeGeneList[i]-1])
    else
      NodeValues.insert(NodeGeneList[i], 0);

  for (size_t i = 0; i < layers.size(); i++)
  {
    for (size_t j =0; j < layers[i].size(); j++)
    {
      int nodeID = layers[i][j];
      if (nodeID == 0)
      {
        for (auto const& x : DirectedGraph)
          NodeValues[x.first] += bias;
      }
      else
      {
        double result = actFn.Fn(NodeValues[nodeID]);
        for (auto const& x : DirectedGraph)
          NodeValues[x.first] += result;
      }
    }
  }

  // Find the output.
  arma::vec output(outputNodeCount);
  for (size_t i = 0; i < output.n_elem; i++)
    output[i] = NodeValues[i+inputNodeCount+1];

  return output;
}

} // namespace neat
} // namespace mlpack

#endif
