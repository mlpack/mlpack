/**
 * @file genome.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Genome class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_GENOME_HPP
#define MLPACK_METHODS_NEAT_GENOME_HPP

#include <mlpack/prereqs.hpp>
#include "gene.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

/**
 * Genome class
 * 
 * @tparam ActivationFunction The activation function to be used.
 */
template <class ActivationFunction>
class Genome
{
 public:
  /**
   * Creates a genome.
   * 
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param actFn The activation function.
   * @param bias The bias of the genome.
   * @param weightMutationRate The probability of a weight mutating.
   * @param weightMutationSize The degree to which the weight will mutate.
   * @param biasMutationRate The probability of a bias mutating.
   * @param biasMutationSize The degree to which the bias will mutate.
   * @param nodeAdditionRate The probability of a new node being added.
   * @param isAcyclic Denotes whether or not the generated network is acyclic.
   */
  Genome(const size_t inputNodeCount,
         const size_t outputNodeCount,
         ActivationFunction& actFn,
         const double bias,
         const double weightMutationRate,
         const double weightMutationSize,
         const double biasMutationRate,
         const double biasMutationSize,
         const double nodeAdditionRate,
         const double connAdditionRate,
         const bool isAcyclic = false);

  /**
   * Loads the input values into the genome.
   */
  void Input(const arma::vec& input);

  /**
   * Calculates the output of the genome.
   */
  arma::vec Output();

  /**
   * Mutates the genome.
   */
  void Mutate();

 private:
  /*
   * A data structure containing IDs for the node genes. It is maintained in the order
   * [bias node, input nodes, output nodes, hidden nodes].
   */
  std::vector<int> NodeGeneList;

  /*
   * A data structure containing the connection genes sorted by global innovation ID.
   */
  std::vector<ConnectionGene> ConnectionGeneList;

  /*
   * A digraph containing connection genes sorted by source ID, and then
   * secondary sorted by target ID.
   */
  std::map<int, std::map<int, ConnectionGene>> DirectedGraph;

  //! Input node count.
  size_t inputNodeCount;

  //! Output node count.
  size_t outputNodeCount;

  //! Hidden node count.
  size_t hiddenNodeCount;

  //! Activation function.
  ActivationFunction actFn;

  //! Boolean indicating if the phenome is acyclic.
  bool isAcyclic;

  //! Bias.
  double bias;

  //! The probability that weight mutation will occur.
  double weightMutationRate;

  //! The degree to which the weight will mutate.
  double weightMutationSize;

  //! The probability that bias mutation will occur.
  double biasMutationRate;

  //! The degree to which bias will mutate.
  double biasMutationSize;

  //! The probability that a new node will be added.
  double nodeAdditionRate;

  //! The probability that a new connection will be added.
  double connAdditionRate;

  //! The next innovation ID to be allotted.
  size_t nextInnovID = 0;

  //! The next node ID to be allotted
  size_t nextNodeID = 0;

  //! The input to the genome.
  arma::vec input;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "genome_impl.hpp"

#endif
