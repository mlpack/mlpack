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
    aSpecies.SortSpecies();

    // Select parents from elite genomes and crossover.
    size_t numElite = floor(aElitePercentage * aSpeciesSize);
    size_t numDrop = floor((aSpeciesSize - numElite) / 2) * 2;  // Make sure even number.
    numElite = aSpeciesSize - numDrop;

    for (size_t i=numElite; i<aSpeciesSize-1; ++i) {
      // Randomly select two parents from elite genomes.
      size_t idx1 = RandInt(0, numElite);
      size_t idx2 = RandInt(0, numElite);

      // Crossover to get two children genomes.
      Genome::CrossoverWeights(aSpecies.aGenomes[idx1], aSpecies.aGenomes[idx2],
                       aSpecies.aGenomes[i], aSpecies.aGenomes[i+1]);
    }

    // Keep the best genome and mutate the rests.
    for (size_t i=1; i<aSpeciesSize; ++i) {
      aSpecies.aGenomes[i].MutateWeightsBiased(aMutateRate, aMutateSize);
    }
  }

  // Evolution of species.
  void Evolve() {
    // Generate initial species at random.
    size_t generation = 0;
    InitSpecies();
    
    // Repeat
    while (generation < aMaxGeneration) {
    	// Evaluate all genomes in the species.
      for (size_t i=0; i<aSpecies.SpeciesSize(); ++i) {
        double fitness = aTask.EvalFitness(aSpecies.aGenomes[i]);
        aSpecies.aGenomes[i].Fitness(fitness);
      }
      aSpecies.SetBestFitness();

    	// Output some information.
      printf("Generation: %zu\tBest fitness: %f\n", generation, aSpecies.BestFitness());

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
  size_t aSpeciesSize;

  // Max number of generation to evolve.
  size_t aMaxGeneration;

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
