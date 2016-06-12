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
#include "population.hpp"
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
    aPopulationSize = params.aPopulationSize;
    aMaxGeneration = params.aMaxGeneration;
    aMutateRate = params.aMutateRate;
    aCrossoverRate = params.aCrossoverRate;
  }

  // Destructor.
  ~CNE() {}

  // Initializing the population of genomes.
  // It can use population's parametric constructor.
  // Besides, adapt its own style of initialization.
  void InitPopulation() {
    aPopulation = Population(aSeedGenome, aPopulationSize);
  }

  // Reproduce the next population. Heart function for CNE !!!
  // Select parents from G(i) based on their fitness.
  // Apply search operators to parents and produce offspring which
  // form G(i + 1).
  void Reproduce() {
    // Sort population by fitness

    // Select two parents by fitness and generate two children

    // Replace two worst by two children

    // Mutate population, keep the best.
    
  }

  // Evolution of population.
  void Evolve() {
    // Generate initial population at random.
    size_t generation = 0;
    InitPopulation();
    
    // Repeat
    while (generation < aMaxGeneration) {
    	// Evaluate all genomes in the population.
      for (size_t i=0; i<aPopulation.PopulationSize(); ++i) {
        double fitness = aTask.EvalFitness(aPopulation.aGenomes[i]);
        aPopulation.aGenomes[i].Fitness() = fitness;
      }
      aPopulation.SetBestFitness();

    	// Output some information.
      printf("Generation: %d\tBest fitness: %f\n", generation, aPopulation.BestFitness());

    	// Reproduce next generation.
      Reproduce();
      ++generation;
    }
  }

 private:
  // Task.
  TaskType aTask;

  // Seed genome. It is used for init population.
  Genome aSeedGenome;

  // Population size.
  size_t aPopulationSize;

  // Population to evolve.
  Population aPopulation;

  // Max number of generation to evolve.
  size_t aMaxGeneration;

  // Mutation rate.
  double aMutateRate;

  // Crossover rate.
  double aCrossoverRate;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_CNE_HPP



