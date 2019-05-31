/**
 * @file gene_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the Gene classes.
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

ConnectionGene::ConnectionGene(const size_t globalInnovationID,
                               const double weight,
                               const size_t source,
                               const size_t target):
    globalInnovationID(globalInnovationID),
    weight(weight),
    source(source),
    target(target)
{ /* Nothing to do here */ }

ConnectionGene::~ConnectionGene()
{ /* Nothing to do here */ }

void ConnectionGene::Mutate(const double mutationSize)
{
  weight += mutationSize * arma::randu<double>();
}

} // namespace neat
} // namespace mlpack

#endif
