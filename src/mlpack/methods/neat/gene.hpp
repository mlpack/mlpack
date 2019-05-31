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
 * Connection gene class.
 */
class ConnectionGene
{
 public:
  /**
   * Creates a connection gene between two node genes.
   *
   * @param globalInnovationID The global innovation ID of the gene.
   * @param weight The weight of the connection.
   * @param source The source of the connection.
   * @param target The target of the connection.
   */
  ConnectionGene(const size_t globalInnovationID,
                 const double weight,
                 const size_t source,
                 const size_t target);

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

  //! Get Source gene.
  size_t getSource() const { return Source; }
  //! Set Source gene.
  size_t& setSource() { return Source; }

  //! Get target gene.
  size_t getTarget() const { return target; }
  //! Set target gene.
  size_t& setTarget() { return target; }

 private:
  //! Global Innovation ID.
  size_t globalInnovationID;

  //! Weight of the connection.
  double weight;

  //! Source gene.
  size_t source;

  //! Target gene.
  size_t target;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "gene_impl.hpp"

#endif
