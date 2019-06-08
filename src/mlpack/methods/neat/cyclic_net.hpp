/**
 * @file cyclic_net.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the CyclicNet class, which represents a cyclic neural
 * network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_CYCLIC_NET_HPP
#define MLPACK_METHODS_NEAT_CYCLIC_NET_HPP

#include <mlpack/prereqs.hpp>
#include "gene.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{
/**
 * A class representation of a cyclic neural network. Genomes are decoded
 * into this class if the `isAcyclic` parameter is set to `false`. It is used
 * to evaluate an input to the genome.
 * 
 * @tparam ActivationFunction The activation function. 
 */
template <class ActivationFunction>
class CyclicNet
{
 public:
  /**
   * Creates a CyclicNet object.
   * 
   * @param DirectedGraph A map of maps storing connection genes, whose first
   *    key is the source ID, and second key is the target ID.
   * @param actFn The activation function.
   * @param nodeCount The number of nodes.
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param timeStepsPerActivation The number of time steps per activation.
   * @param bias The bias.
   */
  CyclicNet(std::map<size_t, std::map<size_t, ConnectionGene>>& directedGraph,
            ActivationFunction& actFn,
            const size_t nodeCount,
            const size_t inputNodeCount,
            const size_t outputNodeCount,
            const size_t timeStepsPerActivation,
            const double bias);

  /**
   * Finds the output of the network from the given input.
   */
  arma::vec Evaluate(arma::vec input);

 private:
  size_t nodeCount;

  /*
   * A digraph containing connection genes sorted by source ID, and then
   * secondary sorted by target ID.
   */
  std::map<size_t, std::map<size_t, ConnectionGene>> directedGraph;

  //! Activation function.
  ActivationFunction& actFn;

  //! Input node count.
  size_t inputNodeCount;

  //! Output node count.
  size_t outputNodeCount;

  //! Node count.
  size_t nodeCount;

  //! The number of time steps per activation.
  const size_t timeStepsPerActivation;

  //! Bias.
  double bias;

  std::vector<double> nodeValues;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "cyclic_net_impl.hpp"

#endif
