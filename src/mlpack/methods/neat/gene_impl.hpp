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
namespace neat /** NeuroEvolution of Augmenting Topologies {

BaseGene::BaseGene(const size_t geneID) : geneID(geneID)
{ /* Nothing to do here */ }

template<class ActivationFunction>
NodeGene<ActivationFunction>::NodeGene(const size_t geneID,
                                       const ActivationFunction& actFn):
    BaseGene(geneID),
    actFn(actFn)
{ /* Nothing to do here */ }

template<class ActivationFunction>
NodeGene<ActivationFunction>::~NodeGene()
{ /* Nothing to do here */ }

template<class ActivationFunction>
double NodeGene<ActivationFunction>::Activate(const arma::vec& input,
                                              const arma::vec& weights)
{
  return actFn.Fn(arma::dot(input,weights));
}

BiasGene::BiasGene(const size_t geneID,
                   const double bias):
    BaseGene(geneID),
    bias(bias)
{ /* Nothing to do here */ }

double BiasGene::Activate()
{
  return bias;
}

void BiasGene::Mutate(const double mutationSize)
{
  bias += mutationSize * arma::randu<double>();
}

ConnectionGene::ConnectionGene(const size_t geneID,
                               const size_t globalInnovationID,
                               const double weight,
                               const BaseGene* origin,
                               const BaseGene* destination):
    BaseGene(geneID),
    globalInnovationID(globalInnovationID),
    weight(weight),
    origin(origin),
    destination(destination)
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
