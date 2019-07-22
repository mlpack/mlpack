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
#include <mlpack/core/util/sfinae_utility.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/dual_tree_kmeans.hpp>
#include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>
#include "genome.hpp"
#include "selection_strategies/rank_selection.hpp"
#include "selection_strategies/tournament_selection.hpp"


namespace mlpack{
namespace neat /** NeuroEvolution of Augmenting Topologies */ {

// This gives us a HasStartingGenome<T, U> type (where U is a function pointer)
// we can use with SFINAE to catch when a type has a StartingGenome(...)
// function.
HAS_MEM_FUNC(StartingGenome, HasStartingGenome);

/**
 * The main class for NeuroEvolution of Augmenting Topologies.
 *
 * @tparam TaskType The task NEAT is to be trained on.
 * @tparam ActivationFunction The activation function used.
 * @tparam SelectionPolicy The policy used to select genomes.
 */
template <class TaskType,
          class ActivationFunction = ann::HardSigmoidFunction,
          class SelectionPolicy = RankSelection>
class NEAT
{
 public:
  /**
   * Creates a NEAT class instance.
   *
   * @param task An instance of the task NEAT is to be trained on.
   * @param inputNodeCount The number of input nodes of the genomes.
   * @param outputNodeCount The number of output nodes of the genomes.
   * @param popSize The size of the population.
   * @param maxGen The maximum number of generations for which NEAT should run.
   * @param initialBias The bias with which genomes are initialized.
   * @param initialWeight The weight of connections with which genomes are 
   *    initialized.
   * @param weightMutationProb The probability of a weight being mutated.
   * @param weightMutationSize The degree of mutation of a weight.
   * @param biasMutationProb The probability of the bias being mutated.
   * @param biasMutationSize The degree of mutation of bias.
   * @param nodeAdditionProb The probability of a new node being added.
   * @param connAdditionProb The probability of a connection being added.
   * @param connDeletionProb The probability of a connection being deleted.
   * @param disableProb The probability of a disabled gene becoming enabled
   *    during crossover.
   * @param elitismProp The proportion of a species that is considered elite.
   * @param finalFitness The desired fitness of the genomes. If it is 0, no
   *    no limit on the fitness is considered.
   * @param complexityThreshold The maximum complexity allowed.
   * @param maxSimplifyGen The maximum number of generations for which
   *    simplification will occur.
   * @param isAcyclic Denotes whether or not the genome is meant to be acyclic.
   */
  NEAT(TaskType& task,
       const size_t inputNodeCount,
       const size_t outputNodeCount,
       const size_t popSize,
       const size_t maxGen,
       const size_t numSpecies,
       const double initialBias = 1,
       const double initialWeight = 0,
       const double weightMutationProb = 0.8,
       const double weightMutationSize = 0.5,
       const double biasMutationProb = 0.7,
       const double biasMutationSize = 0.5,
       const double nodeAdditionProb = 0.2,
       const double connAdditionProb = 0.5,
       const double connDeletionProb = 0.5,
       const double disableProb = 0.2,
       const double elitismProp = 0.1,
       const double finalFitness = 0,
       const size_t complexityThreshold = 0,
       const size_t maxSimplifyGen = 10,
       const bool isAcyclic = false);

  /**
   * Trains the model on the task and returns the best Genome.
   */
  Genome<ActivationFunction> Train();

  /**
   * Performs a single generation of NEAT.
   */
  Genome<ActivationFunction> Step();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  //! Get initial bias.
  double InitialBias() const { return initialBias; }
  //! Set initial bias.
  double& InitialBias() { return initialBias; }

  //! Get initial weight.
  double InitialWeight() const { return initialWeight; }
  //! Set initial weight.
  double& InitialWeight() { return initialWeight; }

  //! Get probability of weight mutation.
  double WeightMutationProb() const { return weightMutationProb; }
  //! Set probability of weight mutation.
  double& WeightMutationProb() { return weightMutationProb; }

  //! Get degree of weight mutation.
  double WeightMutationSize() const { return weightMutationSize; }
  //! Set degree of weight mutation.
  double& WeightMutationSize() { return weightMutationSize; }

  //! Get probability of bias mutation.
  double BiasMutationProb() const { return biasMutationProb; }
  //! Set probability of bias mutation.
  double& BiasMutationProb() { return biasMutationProb; }

  //! Get degree of bias mutation.
  double BiasMutationSize() const { return biasMutationSize; }
  //! Set probability of bias mutation.
  double& BiasMutationSize() { return biasMutationSize; }

  //! Get probability of node addition.
  double NodeAdditionProb() const { return nodeAdditionProb; }
  //! Set probability of node addition.
  double& NodeAdditionProb() { return nodeAdditionProb; }

  //! Get probability of connection addition.
  double ConnAdditionProb() const { return connAdditionProb; }
  //! Set probability of connection addition.
  double& ConnAdditionProb() { return connAdditionProb; }

  //! Get probability that a disabled connection is enabled during crossover.
  double DisableProb() const { return disableProb; }
  //! Set probability that a disabled connection is enabled during crossover.
  double& DisableProb() { return disableProb; }

  //! Get the proportion of a species considered elite.
  double ElitismProp() const { return elitismProp; }
  //! Set the proportion of a species considered elite.
  double& ElitismProp() { return elitismProp; }

  //! Get the desired final fitness.
  double FinalFitness() const { return finalFitness; }
  //! Set the desired final fitness.
  double& FinalFitness() { return finalFitness; }

  //! Get the complexity threshold.
  size_t ComplexityThreshold() const { return complexityThreshold; }
  //! Set the complexity threshold.
  size_t& ComplexityThreshold() { return complexityThreshold; }

  //! Get the maximum number of generations for simplification.
  size_t MaxSimplifyGen() const { return maxSimplifyGen; }
  //! Set the maximum number of generations for simplification.
  size_t MaxSimplifyGen() { return maxSimplifyGen; }

  //! Get the current complexity ceiling.
  size_t CurrentComplexityCeiling() const { return currentComplexityCeiling; }
  //! Set the current complexity ceiling.
  size_t& CurrentComplexityCeiling() { return currentComplexityCeiling; }

  //! Get the mean complexity.
  double MeanComplexity() const { return meanComplexity; }
  //! Set the mean complexity.
  double& MeanComplexity() { return meanComplexity; }

  //! Get the boolean denoting if the genome is acyclic or not.
  bool IsAcyclic() const { return isAcyclic; }
  //! Set the boolean denoting if the genome is acyclic or not.
  bool& IsAcyclic() { return isAcyclic; }

  //! Get the starting genome.
  Genome<ActivationFunction> StartingGenome() const { return startingGenome; }
  //! Set the starting genome.
  Genome<ActivationFunction>& StartingGenome() { return startingGenome; }

  //! Get the number of contenders for tournament selection.
  size_t ContenderNum() const { return contenderNum; }
  //! Set the number of contenders for tournament selection.
  size_t& ContenderNum() { return contenderNum; }

  //! Get the base probability for tournament selection.
  double TournamentSelectProb() const { return tournamentSelectProb; }
  //! Set the base probability for tournament selection.
  double& TournamentSelectProb() { return tournamentSelectProb; }
 private:
  // Crosses over two genomes.
  Genome<ActivationFunction> Crossover(Genome<ActivationFunction>& gen1,
                                       Genome<ActivationFunction>& gen2);

  // Creates the next generation through reproduction.
  void Reproduce();

  /**
   * Speciates the population. If init is true, it performs the first
   * speciation without knowledge of centroids.
   */
  void Speciate(bool init);

  // Compares genome based on fitness.
  static bool compareGenome(Genome<ActivationFunction>& gen1,
                            Genome<ActivationFunction>& gen2);

  /**
   * Creates the initial genome population. This function is called if the
   * StartingGenome() function exists.
   */
  template <typename Task = TaskType>
  typename std::enable_if<
      HasStartingGenome<Task, std::vector<ConnectionGene>(Task::*)()>::value,
      void>::type
  Initialize();

  /**
   * Creates the initial genome population. This function is called if the
   * StartingGenome() function does not exist.
   */
  template <typename Task = TaskType>
  typename std::enable_if<
      !HasStartingGenome<Task, std::vector<ConnectionGene>(Task::*)()>::value,
      void>::type
  Initialize();

  /**
   * Selection in the case of TournamentSelection.
   */
  template <typename Policy = SelectionPolicy>
  typename std::enable_if<
      std::is_same<Policy, TournamentSelection>::value, void>::type
  Select(arma::vec& fitnesses,
         arma::uvec& selection,
         const size_t contenderNum,
         const double prob);

  /**
   * Selection for other policies.
   */
  template <typename Policy = SelectionPolicy>
  typename std::enable_if<
      !std::is_same<Policy, TournamentSelection>::value, void>::type
  Select(arma::vec& fitnesses,
         arma::uvec& selection,
         const size_t /* Unused */,
         const double /* Unused */);

  //! The task that the model is trained on.
  TaskType task;

  //! The list of genomes in the population.
  std::vector<Genome<ActivationFunction>> genomeList;

  //! The list of species, each containing a list of genomes.
  std::vector<std::vector<Genome<ActivationFunction>>> speciesList;

  //! The centroids of the genome clusters.
  arma::mat centroids;

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
  double initialBias;

  //! The initial weights of the connections.
  double initialWeight;

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

  //! The probability of a connection being deleted.
  double connDeletionProb;

  /**
   * The probability that an inherited gene is disabled if either of it's
   * parents are disabled.
   */
  double disableProb;

  //! The proportion of a species that is considered elite.
  double elitismProp;

  //! The desired final fitness.
  double finalFitness;

  //! The maximum complexity after which NEAT performes simplification.
  size_t complexityThreshold;

  //! The current complexity ceiling.
  size_t currentComplexityCeiling;

  //! The mean population complexity.
  double meanComplexity;

  //! The generation at which transition in strategy occurred.
  size_t lastTransitionGen;

  //! The maximum number of generations for simplification.
  size_t maxSimplifyGen;

  /**
   * The search mode. If it is 1, NEAT is simplifying. If it is 0, NEAT is
   * complexifying.
   */   
  size_t searchMode;

  //! Denotes whether or not the genome is meant to be cyclic.
  bool isAcyclic;

  //! The starting genome.
  Genome<ActivationFunction> startingGenome;

  //! The number of contenders in tournament selection.
  size_t contenderNum;

  //! The base probability used in tournament selection.
  double tournamentSelectProb;
};

} // namespace neat
} // namespace mlpack

// Include implementation.
#include "neat_impl.hpp"

#endif
