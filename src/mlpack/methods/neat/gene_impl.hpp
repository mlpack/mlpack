/**
 * @file gene_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the ConnectionGene class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_GENE_IMPL_HPP
#define MLPACK_METHODS_NEAT_GENE_IMPL_HPP

// In case it hasn't yet been included.
#include "gene.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */{

// Creates a connection gene object.
ConnectionGene::ConnectionGene(const size_t globalInnovationID,
                               const double weight,
                               const size_t source,
                               const size_t target,
                               const bool enabled):
    globalInnovationID(globalInnovationID),
    weight(weight),
    source(source),
    target(target),
    enabled(enabled)
{ /* Nothing to do here */ }

// Default destructor for a connection gene.
ConnectionGene::~ConnectionGene()
{ /* Nothing to do here */ }

// Default constructor for a connection gene.
ConnectionGene::ConnectionGene():
    globalInnovationID(0),
    weight(0),
    source(0),
    target(0),
    enabled(false)
{ /* Nothing to do here */ }

void ConnectionGene::Mutate(const double mutationSize)
{
  weight += mutationSize * arma::randn();
}

template <typename Archive>
void ConnectionGene::serialize(Archive& ar,
                              const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(globalInnovationID);
  ar & BOOST_SERIALIZATION_NVP(weight);
  ar & BOOST_SERIALIZATION_NVP(source);
  ar & BOOST_SERIALIZATION_NVP(target);
  ar & BOOST_SERIALIZATION_NVP(enabled);
}

} // namespace neat
} // namespace mlpack

#endif
