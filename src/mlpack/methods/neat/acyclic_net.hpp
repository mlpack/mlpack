/**
 * @file acyclic_net.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the AcyclicNet class, which represents an acyclic neural
 * network.
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
/**
 * A class representation of an acyclic neural network. Genomes are decoded
 * into this class if the `isAcyclic` parameter is set to `true`. It is used
 * to evaluate an input to the genome.
 * 
 * The steps of the decoding and activating are as follows:
 * 1. The nodes are each assigned a "depth" depending on the number of "jumps"
 *    it takes to reach it from an input node. If there are two paths to the
 *    node, the longer path is used to find the depth.
 * 2. Based on the depth, the nodes are divided into layers.
 * 3. Starting from layer zero, the layers are activated one by one.
 * 4. The output is taken from the last layer, i.e. the output nodes.
 * 
 * @tparam ActivationFunction The activation function. 
 */
template <class ActivationFunction>
class AcyclicNet
{
 public:
  /**
   * Creates an AcyclicNet object.
   *
   * @param nodeCount The number of nodes.
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param bias The bias.
   */
  AcyclicNet(const size_t nodeCount,
             const size_t inputNodeCount,
             const size_t outputNodeCount,
             const double bias);

  /**
   * Finds the output of the network from the given input.
   */
  void Evaluate(const arma::vec& input,
                arma::vec& output,
                std::map<size_t, std::map<size_t, ConnectionGene>>&
                    directedGraph,
                const std::vector<size_t>& nodeDepths);
 private:
  //! The number of nodes.
  size_t nodeCount;

  //! Input node count.
  size_t inputNodeCount;

  //! Output node count.
  size_t outputNodeCount;

  //! Bias.
  double bias;

  //! A data structure storing the nodes by layer.
  std::vector<std::vector<size_t>> layers;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "acyclic_net_impl.hpp"

#endif
