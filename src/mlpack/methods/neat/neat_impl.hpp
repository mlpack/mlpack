/**
 * @file neat_impl.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Implementation of the NEAT class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NEAT_NEAT_IMPL_HPP
#define MLPACK_METHODS_NEAT_NEAT_IMPL_HPP

// In case it hasn't been included.
#include "neat.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

template <class TaskType,
          class ActivationFunction>
NEAT<TaskType, ActivationFunction>::NEAT(TaskType& task,
                                         ActivationFunction& actFn,
                                         const size_t inputNodeCount,
                                         const size_t outputNodeCount,
                                         const size_t popSize,
                                         const double bias,
                                         const double weightMutationProb,
                                         const double weightMutationSize,
                                         const double biasMutationProb,
                                         const double biasMutationSize,
                                         const double nodeAdditionProb,
                                         const double connAdditionProb,
                                         const bool isAcyclic):
    task(task),
    actFn(actFn),
    inputNodeCount(inputNodeCount),
    outputNodeCount(outputNodeCount),
    popSize(popSize),
    bias(bias),
    weightMutationProb(weightMutationProb),
    weightMutationSize(weightMutationSize),
    biasMutationProb(biasMutationProb),
    biasMutationSize(biasMutationSize),
    nodeAdditionProb(nodeAdditionProb),
    connAdditionProb(connAdditionProb),
    isAcyclic(isAcyclic)
{ /* Nothing to do here yet */ }

Genome Crossover(Genome& gen1, Genome& gen2)
{
  
}

} // namespace neat
} // namespace mlpack

#endif
