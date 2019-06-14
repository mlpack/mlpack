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
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/dual_tree_kmeans.hpp>
#include "selection_strategies/rank_selection.hpp"
#include "selection_strategies/roulette_selection.hpp"
#include "selection_strategies/tournament_selection.hpp"

namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

template <class TaskType,
          class ActivationFunction,
          class SelectionPolicy>
class NEAT
{
 public:
  NEAT(TaskType& task,
       const size_t inputNodeCount,
       const size_t outputNodeCount,
       const size_t popSize,
       const size_t maxGen,
       const size_t numSpecies,
       const double bias,
       const double weightMutationProb,
       const double weightMutationSize,
       const double biasMutationProb,
       const double biasMutationSize,
       const double nodeAdditionProb,
       const double connAdditionProb,
       const double disableProb,
       const double elitismProp,
       const bool isAcyclic = false);

  /**
   * Trains the model on the task and returns the best Genome.
   */
  Genome<ActivationFunction> Train();

 private:
  // Crosses over two genomes.
  Genome<ActivationFunction> Crossover(Genome<ActivationFunction>& gen1,
                                       Genome<ActivationFunction>& gen2);

  // Creates the next generation through reproduction.
  void Reproduce();

  // Speciates the population. If init is true, it performs the first
  // speciation without knowledge of centroids.
  void Speciate(bool init);

  static bool compareGenome(Genome<ActivationFunction>& gen1,
                            Genome<ActivationFunction>& gen2);

  // The list of genomes in the population.
  std::vector<Genome<ActivationFunction>> genomeList;

  // The list of species, each containing a list of genomes.
  std::vector<std::vector<Genome<ActivationFunction>>> speciesList;

  // The centroids of the genome clusters.
  arma::mat centroids;

  //! The provided TaskType class that evaluates fitness of the genome.
  TaskType task;

  //! The number of input nodes.
  size_t inputNodeCount;

  //! The number of output nodes.
  size_t outputNodeCount;

  //! The size of the population.
  size_t popSize;

  //! The maximum number of generations.
  size_t maxGen;

  //! The number of species.
  size_t numSpecies;

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

  /**
   * The probability that an inherited gene is disabled if either of it's
   * parents are disabled.
   */
  double disableProb;

  //! The proportion of a species that is considered elite.
  double elitismProp;

  //! Denotes whether or not the genome is meant to be cyclic.
  bool isAcyclic;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "neat_impl.hpp"

#endif
