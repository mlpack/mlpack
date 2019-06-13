/**
 * @file gene.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the ConnectionGene class.
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
 * A class that represents a connection gene in a genome.
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
   * @param enabled Denotes whether the connection is enabled or not.
   */
  ConnectionGene(const size_t globalInnovationID,
                 const double weight,
                 const size_t source,
                 const size_t target,
                 const bool enabled = true);

  /**
   * Destroys the connection gene.
   */
  ~ConnectionGene();

  /**
   * Mutate the weights of the gene.
   *
   * @param mutationSize The strength of mutation noise to be added.
   */
  void Mutate(const double mutationSize);

  //! Get global innovation ID.
  size_t getInnovationID() const { return globalInnovationID; }
  //! Set global innovation ID.
  size_t& setInnovationID() { return globalInnovationID; }

  //! Get connection weight.
  double getWeight() const { return weight; }
  //! Set connection weight.
  double& setWeight() { return weight; }

  //! Get Source gene.
  size_t getSource() const { return source; }
  //! Set Source gene.
  size_t& setSource() { return source; }

  //! Get target gene.
  size_t getTarget() const { return target; }
  //! Set target gene.
  size_t& setTarget() { return target; }

  //! Check if the connection is enabled.
  bool isEnabled() const { return enabled; }
  //! Enable or disable the connection.
  bool& setEnabled() { return enabled; }

 private:
  //! Global Innovation ID.
  size_t globalInnovationID;

  //! Weight of the connection.
  double weight;

  //! Source gene.
  size_t source;

  //! Target gene.
  size_t target;

  //! Boolean denoting whether or not the connection is enabled.
  bool enabled;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "gene_impl.hpp"

#endif
