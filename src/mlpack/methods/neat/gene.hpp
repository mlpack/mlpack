/**
 * @file gene.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the Gene classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_GENE_HPP
#define MLPACK_METHODS_NEAT_GENE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

/**
 * Base Gene class.
 */
class BaseGene
{
 public:
  /**
   * Creates a base gene object.
   * 
   * @param geneID ID of the gene.
   */
  BaseGene(const size_t geneID);

  //! Get the gene ID.
  size_t getGeneID() const { return geneID; }
  //! Set the gene ID.
  size_t& setGeneID() { return geneID; }
 
 private:
  //! The ID of the gene. 
  size_t geneID;
};

/**
 * Node gene class.
 * 
 * @tparam ActivationFunction The activation function to use.
 */
template<class ActivationFunction>
class NodeGene : public BaseGene
{
 public:
  /**
   * Creates a node gene.
   * 
   * @param geneID ID of the gene.
   * @param actFn The activation function used by the gene.
   */ 
  NodeGene(const size_t geneID,
           const ActivationFunction& actFn);

  /**
   * Destroys the node gene.
   */
  ~NodeGene();

  /**
   * Computes the output of the node gene.
   * 
   * @param input The input of the node gene.
   * @param weights The weights of the connection genes connected to this
   *     node gene.
   */
  double Activate(const arma::vec& input,
                  const arma::vec& weights);

 private:
  //! Activation function.
  ActivationFunction actFn;
};

/**
 * Bias gene class.
 */
class BiasGene : public BaseGene
{
 public:
  /**
   * Creates a bias gene.
   * 
   * @param geneID ID of the gene.
   * @param bias The bias.
   */
  BiasGene(const size_t geneID,
           const double bias);

  /**
   * Computes the output of the bias gene, which is it's bias.
   */
  double Activate();

  /*
   * Mutate the bias of the gene.
   */
  void Mutate(const double mutationSize);

  //! Get the bias.
  double getBias() const { return bias; }
  //! Set the bias.
  double& setBias() { return bias; }

  //! Get the global innovation ID. It is always zero.
  double getGlobalInnovationID() const { return globalInnovationID; }

 private:
  //! Global innovation ID. It is always zero.
  const size_t globalInnovationID = 0;

  //! Bias.
  double bias;
};

/**
 * Connection gene class.
 */
class ConnectionGene : public BaseGene
{
 public:
  /**
   * Creates a connection gene between two node genes.
   * 
   * @param geneID ID of the gene.
   * @param globalInnovationID The global innovation ID of the gene.
   * @param weight The weight of the connection.
   * @param origin The origin of the connection.
   * @param destination The destination of the connection.
   */
  ConnectionGene(const size_t geneID,
                 const size_t globalInnovationID,
                 const double weight,
                 const BaseGene* origin,
                 const BaseGene* destination);

  /**
   * Destroys the connection gene.
   */
  ~ConnectionGene();

  /*
   * Mutate the weights of the gene.
   * 
   * @param mutationSize The strength of mutation noise to be added.
   */
  void Mutate(const double mutationSize);

  //! Get global innovation ID.
  size_t getGlobalInnovationID() const { return globalInnovationID; }
  //! Set global innovation ID.
  size_t& setGlobalInnovationID() { return globalInnovationID; }

  //! Get connection weight.
  double getWeight() const { return weight; }
  //! Set connection weight.
  double& setWeight() { return weight; }

  //! Get origin gene.
  BaseGene* getOrigin() const { return origin; }
  //! Set origin gene.
  BaseGene*& setOrigin() { return origin; }

  //! Get destination gene.
  BaseGene* getOrigin() const { return destination; }
  //! Set destination gene.
  BaseGene*& setOrigin() { return destination; }

 private:
  //! Global Innovation ID.
  size_t globalInnovationID;

  //! Weight of the connection.
  double weight;

  //! Origin gene.
  BaseGene* origin;

  //! Destination gene.
  BaseGene* destination;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "gene_impl.hpp"

#endif
