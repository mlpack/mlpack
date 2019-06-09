/**
 * @file neat.hpp
 * @author Rahul Ganesh Prabhu
 *
 * Definition of the NEAT class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NEAT_NEAT_HPP
#define MLPACK_METHODS_NEAT_NEAT_HPP

#include <mlpack/prereqs.hpp>
#include "genome.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

template <class TaskType,
          class ActivationFunction>
class NEAT
{
 public:
  NEAT(TaskType& task,
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
       const bool isAcyclic = false);

  /**
   * Trains the model on the task and returns the best Genome.
   */
  Genome Train();

 private:

  Genome Crossover(Genome& gen1, Genome& gen2);

  void Reproduce();

  void Speciate();

  std::vector<Genome> genomeList;

  //! The provided TaskType class that evaluates fitness of the genome.
  TaskType task;

  //! The activation function.
  ActivationFunction actFn;

  //! The number of input nodes.
  size_t inputNodeCount;

  //! The number of output nodes.
  size_t outputNodeCount;

  //! The size of the population.
  size_t popSize;

  //! The bias of the networks.
  double bias;

  //! The probability of a connection weight mutating.
  double weightMutationProb;

  //! The degree to which a connection weight will mutate.
  double weightMutationSize;

  //! The probability of the bias mutating.
  double biasMutationProb;

  //! The degree to which the bias will mutate.
  double biasMutationSize;

  //! The probability of a new node being added.
  double nodeAdditionProb;

  //! The probability of a new connection being added.
  double connAdditionProb;

  //! Denotes whether or not the genome is meant to be cyclic.
  bool isAcyclic;
};


} // namespace neat
} // namespace mlpack

// Include implementation.
#include "neat_impl.hpp"

#endif
