/**
 * @file genome.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Genome class which represents a genome in the population.
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
 * A class representation of a genome in the population. The genome is a
 * collection of connection genes which can be decoded into a neural
 * network and be used to evaluate an input.
 * 
 * It also stores the nodes in the form of IDs:
 * Bias node : ID = 0
 * Input nodes : ID = 1 - inputNodeCount
 * Output nodes : ID = inputNodeCount + 1 - inputNodeCount+outputNodeCount
 * Hidden nodes : ID > inputNodeCount + outputNodeCount
 * 
 * The genome can undergo two types of structural mutations:
 * 1. A connection can be split into two, creating a new node.
 * 2. A new connection can be created between two nodes.
 * 
 * @tparam ActivationFunction The activation function to be used.
 */
template <class ActivationFunction = ann::HardSigmoidFunction>
class Genome
{
 public:
  /**
   * Creates a genome. Used during initialization.
   * 
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param bias The bias of the genome.
   * @param weightMutationProb The probability of a weight mutating.
   * @param weightMutationSize The degree to which the weight will mutate.
   * @param biasMutationProb The probability of a bias mutating.
   * @param biasMutationSize The degree to which the bias will mutate.
   * @param nodeAdditionProb The probability of a new node being added.
   * @param isAcyclic Denotes whether or not the generated network is acyclic.
   */
  Genome(const size_t inputNodeCount,
         const size_t outputNodeCount,
         const double bias,
         const double initialWeight,
         const double weightMutationProb,
         const double weightMutationSize,
         const double biasMutationProb,
         const double biasMutationSize,
         const double nodeAdditionProb,
         const double connAdditionProb,
         const double connDeletionProb,
         const double nodeDeletionProb = 0,
         const bool isAcyclic = false);

  /**
   * Creates a genome. Used during cyclic reproduction.
   * 
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param connectionGeneList A vector of connection genes sorted by
   *     innovation ID.
   * @param nextNodeID The number of nodes in the system.
   * @param bias The bias of the genome.
   * @param weightMutationProb The probability of a weight mutating.
   * @param weightMutationSize The degree to which the weight will mutate.
   * @param biasMutationProb The probability of a bias mutating.
   * @param biasMutationSize The degree to which the bias will mutate.
   * @param nodeAdditionProb The probability of a new node being added.
   * @param isAcyclic Denotes whether or not the generated network is acyclic.
   */
  Genome(std::vector<ConnectionGene>& connectionGeneList,
         const size_t inputNodeCount,
         const size_t outputNodeCount,
         const size_t nextNodeID,
         const double bias,
         const double weightMutationProb,
         const double weightMutationSize,
         const double biasMutationProb,
         const double biasMutationSize,
         const double nodeAdditionProb,
         const double connAdditionProb,
         const double connDeletionProb,
         const double nodeDeletionProb = 0,
         const bool isAcyclic = false);

  /**
   * Creates a genome. Used during acyclic reproduction.
   * 
   * @param inputNodeCount The number of input nodes.
   * @param outputNodeCount The number of output nodes.
   * @param connectionGeneList A vector of connection genes sorted by
   *     innovation ID.
   * @param nodeDepths A vector of node depths, where the depth is the maximum
   *     number of "jumps" it takes to get to a node from an input node.
   * @param nextNodeID The number of nodes in the system.
   * @param bias The bias of the genome.
   * @param weightMutationProb The probability of a weight mutating.
   * @param weightMutationSize The degree to which the weight will mutate.
   * @param biasMutationProb The probability of a bias mutating.
   * @param biasMutationSize The degree to which the bias will mutate.
   * @param nodeAdditionProb The probability of a new node being added.
   * @param isAcyclic Denotes whether or not the generated network is acyclic.
   */
  Genome(std::vector<ConnectionGene>& connectionGeneList,
         std::vector<size_t>& nodeDepths,
         const size_t inputNodeCount,
         const size_t outputNodeCount,
         const size_t nextNodeID,
         const double bias,
         const double weightMutationProb,
         const double weightMutationSize,
         const double biasMutationProb,
         const double biasMutationSize,
         const double nodeAdditionProb,
         const double connAdditionProb,
         const double connDeletionProb,
         const double nodeDeletionProb = 0,
         const bool isAcyclic = false);

  /**
   * Calculates the output of the genome based on the input.
   */
  arma::vec Evaluate(arma::vec& input);

  /**
   * Mutates the genome.
   */
  void Mutate();

  /**
   * Returns the parameters of the genome in the form of an adjacency matrix,
   * where the row numbers denote the IDs of the source neurons and the column
   * numbers denote the IDs of the target neurons, and the values are the
   * weights of the enabled connections.
   */
  arma::mat Parameters();

  /**
   * Prints genome (Used for debugging purposes).
   */
  void Print();

  /**
   * Calculates complexity
   */
  size_t Complexity();

  /**
   * A data structure containing the connection genes sorted by global
   * innovation ID.
   */
  std::vector<ConnectionGene> connectionGeneList;

  /**
   * A vector of node depths, where the depth is the maximum number of "jumps"
   * it takes to get to a node from an input node. This is only used in acyclic
   * cases.
   */
  std::vector<size_t> nodeDepths;

  /**
   * The buffer of added connections in this generation.
   */
  static std::map<std::pair<size_t, size_t>, size_t> mutationBuffer;

  /**
   * A digraph containing connection genes sorted by source ID, and then
   * secondary sorted by target ID.
   */
  std::map<size_t, std::map<size_t, ConnectionGene>> directedGraph;

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Get fitness.
  double Fitness() const { return fitness; }
  //! Set fitness.
  double& Fitness() { return fitness; }

  //! Get node count.
  size_t getNodeCount() const { return nextNodeID; }

  //! Get input node count.
  size_t getInputNodeCount() const { return inputNodeCount; }

  //! Get output node count.
  size_t getOutputNodeCount() const { return outputNodeCount; }

  //! Get node addition probability.
  double NodeAdditionProb() const { return nodeAdditionProb; }
  //! Set node addition probability.
  double& NodeAdditionProb() { return nodeAdditionProb; }

  //! Get connection addition probability.
  double ConnAdditionProb() const { return connAdditionProb; }
  //! Set connection addition probability.
  double& ConnAdditionProb() { return connAdditionProb; }

  //! Get connection deletion probability.
  double ConnDeletionProb() const { return connDeletionProb; }
  //! Set connection deletion probability.
  double& ConnDeletionProb() { return connDeletionProb; }

  //! Get node deletion probability.
  double NodeDeletionProb() const { return nodeDeletionProb; }
  //! Set node deletion probability,
  double& NodeDeletionProb() { return nodeDeletionProb; }

  //! Get bias.
  double Bias() const { return bias; }
  //! Set bias.
  double& Bias() { return bias; }

  //! The next innovation ID to be allotted.
  static size_t nextInnovID;
 private:
  /**
   * Stores the activation of the nodes. Only used if isAcyclic is set to false.
   */
  std::vector<double> outputNodeValues;

  /**
   * A recursive function that assigns depth to nodes. Only used in acyclic
   * cases.
   */
  void Traverse(size_t startID);

  void AddConnMutation();

  void AddNodeMutation();

  void DelConnMutation();

  void DelNodeMutation();

  //! Input node count.
  size_t inputNodeCount;

  //! Output node count.
  size_t outputNodeCount;

  //! The next node ID to be allotted.
  size_t nextNodeID;

  //! Bias.
  double bias;

  //! Initial Weight.
  double initialWeight;

  //! The probability that weight mutation will occur.
  double weightMutationProb;

  //! The degree to which the weight will mutate.
  double weightMutationSize;

  //! The probability that bias mutation will occur.
  double biasMutationProb;

  //! The degree to which bias will mutate.
  double biasMutationSize;

  //! The probability that a new node will be added.
  double nodeAdditionProb;

  //! The probability that a new connection will be added.
  double connAdditionProb;

  //! The probability that a connection will be deleted.
  double connDeletionProb;

  //! The probability that a node will be deleted.
  double nodeDeletionProb;

  //! The fitness.
  double fitness;

  //! Boolean indicating if the phenome is acyclic.
  bool isAcyclic;

  //! Stuff for debugging.
  arma::vec inputt;
  arma::vec output;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "genome_impl.hpp"

#endif
