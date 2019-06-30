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
   * @param nodeCount The number of nodes.
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param bias The bias.
   */
  CyclicNet(const size_t nodeCount,
            const size_t inputNodeCount,
            const size_t outputNodeCount,
            const double bias);

  /**
   * Finds the output of the network from the given input.
   */
  void Evaluate(arma::vec& input,
                arma::vec& output,
                std::vector<double>& outputNodeValues,
                std::map<size_t, std::map<size_t, ConnectionGene>>& directedGraph);

 private:
  //! Node count.
  size_t nodeCount;

  //! Input node count.
  size_t inputNodeCount;

  //! Output node count.
  size_t outputNodeCount;

  //! Bias.
  double bias;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "cyclic_net_impl.hpp"

#endif
