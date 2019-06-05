/**
 * @file acyclic_net.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Acyclic net classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_ACYCLIC_NET_HPP
#define MLPACK_METHODS_NEAT_ACYCLIC_NET_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

template <class ActivationFunction>
class AcyclicNet
{
 public:
  /**
   * Creates an AcyclicNet object.
   * 
   * @param NodeGeneList A vector storing all the node IDs.
   * @param DirectedGraph A map of maps storing connection genes, whose first
   *    key is the source ID, and second key is the target ID.
   * @param actFn The activation function.
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param bias The bias.s
   */
  AcyclicNet(std::vector<int>& NodeGeneList,
             std::map<int, std::map<int, ConnectionGene>>& DirectedGraph,
             ActivationFunction& actFn,
             const size_t inputNodeCount,
             const size_t outputNodeCount,
             const double bias);

  /**
   * Finds the output of the network from the given input.
   */
  arma::vec Evaluate(arma::vec input);

 private:
  std::vector<int> NodeGeneList;

  std::map<int, std::map<int, ConnectionGene>> DirectedGraph;

  ActivationFunction& actFn;

  size_t inputNodeCount;

  size_t outputNodeCount;

  double bias;

  std::map<size_t, size_t> nodeDepths;

  std::vector<std::vector<size_t>> layers;

  void TraverseNode(size_t nodeID, size_t depth);
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "acyclic_net_impl.hpp"

#endif
