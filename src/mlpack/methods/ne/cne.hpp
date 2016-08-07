 /**
 * @file cne.hpp
 * @author Bang Liu
 *
 * Definition of CNE class.
 */
#ifndef MLPACK_METHODS_NE_CNE_HPP
#define MLPACK_METHODS_NE_CNE_HPP

#include <cstddef>
#include <cstdio>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"
#include "species.hpp"
#include "tasks.hpp"
#include "parameters.hpp"

namespace mlpack {
namespace ne {

/**
 * This class implements Conventional Neuro-evolution (CNE): weight
 * evolution on topologically fixed neural networks.
 */
template<typename TaskType>
class CNE {
 public:
  // Parametric constructor.
  CNE(TaskType task, Genome& seedGenome, Parameters& params) {
    aTask = task;
    aSeedGenome = seedGenome;
    aSpeciesSize = params.aSpeciesSize;
    aMaxGeneration = params.aMaxGeneration;
    aMutateRate = params.aMutateRate;
    aMutateSize = params.aMutateSize;
    aElitePercentage = params.aElitePercentage;
  }

  // Destructor.
  ~CNE() {}

  // Soft mutation: add a random value chosen from
  // initialization prob distribution with probability p.
  // TODO: here we use uniform distribution. Can we use exponential distribution?
  static void MutateWeightsBiased(Genome& genome, double mutateProb, double mutateSize) {
    for (int i=0; i<genome.aLinkGenes.size(); ++i) {
      double p = mlpack::math::Random();  // rand 0~1
      if (p < mutateProb) {
        double deltaW = mlpack::math::RandNormal(0, mutateSize);
        double oldW = genome.aLinkGenes[i].Weight();
        genome.aLinkGenes[i].Weight(oldW + deltaW);
      }
    }
  }

  // Hard mutation: replace with a random value chosen from
  // initialization prob distribution with probability p.
  // TODO: here we use uniform distribution. Can we use exponential distribution?
  static void MutateWeightsUnbiased(Genome& genome, double mutateProb, double mutateSize) {
    for (int i=0; i<genome.aLinkGenes.size(); ++i) {
      double p = mlpack::math::Random();
      if (p < mutateProb) {
        double weight = mlpack::math::RandNormal(0, mutateSize);
        genome.aLinkGenes[i].Weight(weight);
      }
    }
  }

  // Randomly select weights from one parent genome.
  // NOTICE: child genomes need to set genome id based on its population's max id.
  static void CrossoverWeights(Genome& momGenome, 
                               Genome& dadGenome, 
                               Genome& child1Genome, 
                               Genome& child2Genome) {
    child1Genome = momGenome;
    child2Genome = dadGenome;
    for (int i=0; i<momGenome.aLinkGenes.size(); ++i) { // assume genome are the same structure.
      double t = mlpack::math::RandNormal();
      if (t>0) {  // prob = 0.5
        child1Genome.aLinkGenes[i].Weight(momGenome.aLinkGenes[i].Weight());
        child2Genome.aLinkGenes[i].Weight(dadGenome.aLinkGenes[i].Weight());
      } else {
        child1Genome.aLinkGenes[i].Weight(dadGenome.aLinkGenes[i].Weight());
        child2Genome.aLinkGenes[i].Weight(momGenome.aLinkGenes[i].Weight());
      }
    }
  }

  // Initializing the species of genomes.
  // It can use species's parametric constructor.
  // Besides, adapt its own style of initialization.
  void InitSpecies() {
    aSpecies = Species(aSeedGenome, aSpeciesSize);
  }

  // Reproduce the next species. Heart function for CNE !!!
  // Select parents from G(i) based on their fitness.
  // Apply search operators to parents and produce offspring which
  // form G(i + 1).
  void Reproduce() {
    // Sort species by fitness
    aSpecies.SortGenomes();

    // Select parents from elite genomes and crossover.
    int numElite = floor(aElitePercentage * aSpeciesSize);
    int numDrop = floor((aSpeciesSize - numElite) / 2) * 2;  // Make sure even number.
    numElite = aSpeciesSize - numDrop;
    for (int i=numElite; i<aSpeciesSize-1; ++i) {
      // Randomly select two parents from elite genomes.
      int idx1 = RandInt(0, numElite);
      int idx2 = RandInt(0, numElite);

      // Crossover to get two children genomes.
      CrossoverWeights(aSpecies.aGenomes[idx1], aSpecies.aGenomes[idx2],
                               aSpecies.aGenomes[i], aSpecies.aGenomes[i+1]);
    }

    // Keep the best genome and mutate the rests.
    for (int i=1; i<aSpeciesSize; ++i) {
      MutateWeightsBiased(aSpecies.aGenomes[i], aMutateRate, aMutateSize);
    }
  }

  // Evolution of species.
  void Evolve() {
    // Generate initial species at random.
    int generation = 0;
    InitSpecies();
    
    // Repeat
    while (generation < aMaxGeneration) {
    	// Evaluate all genomes in the species.
      for (int i=0; i<aSpecies.SpeciesSize(); ++i) {
        double fitness = aTask.EvalFitness(aSpecies.aGenomes[i]);
        aSpecies.aGenomes[i].Fitness(fitness);
      }
      aSpecies.SetBestFitnessAndGenome();

    	// Output some information.
      printf("Generation: %zu\tBest fitness: %f\n", generation, aSpecies.BestFitness());
      if (aTask.Success()) {
        printf("Task succeed in %zu iterations.\n", generation);
        exit(0);
      }

    	// Reproduce next generation.
      Reproduce();
      ++generation;
    }
  }

 private:
  // Task.
  TaskType aTask;

  // Seed genome. It is used for init species.
  Genome aSeedGenome;

  // Species to evolve.
  Species aSpecies;

  // Species size.
  int aSpeciesSize;

  // Max number of generation to evolve.
  int aMaxGeneration;

  // Mutation rate.
  double aMutateRate;

  // Mutate size. For normal distribution, it is mutate variance.
  double aMutateSize;

  // Elite percentage.
  double aElitePercentage;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_CNE_HPP
