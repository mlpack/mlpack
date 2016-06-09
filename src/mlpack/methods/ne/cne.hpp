 /**
 * @file cne.hpp
 * @author Bang Liu
 *
 * Definition of CNE class.
 */
#ifndef MLPACK_METHODS_NE_POPULATION_HPP
#define MLPACK_METHODS_NE_POPULATION_HPP

#include <cstddef>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"
#include "population.hpp"
#include "tasks.hpp"

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

  // Destructor.
  ~CNE() {}

  // Initializing the population of genomes.
  void InitPopulation() {

  }

  // Reproduce the next population. Heart function for CNE !!!
  // Select parents from G(i) based on their fitness.
  // Apply search operators to parents and produce offspring which
  // form G(i + 1).
  void Reproduce() {
    
  }

  // Evolution of population.
  void Evolve(size_t maxGeneration, TaskType& task) {
    // Generate initial population at random.
    size_t generation = 0;
    InitPopulation();
    
    // Repeat
    while (generation < maxGeneration) {
    	// Evaluate all genomes in the population.
      for (size_t i=0; i<aPopulation.NumGenome(); ++i) {
        double fitness = task.EvalFitness(aPopulation.aGenomes[i]);
        aPopulation.aGenomes[i].Fitness() = fitness;
      }
      aPopulation.SetBestFitness();

    	// Output some information.
      printf("Best fitness: %f\n", aPopulation.BestFitness())

    	// Reproduce next generation.
      Reproduce();
      ++generation;
    }
  }

 private:
  TaskType aTask;
  Genome aSeedGenome;
  Population aPopulation;

  size_t aPopulationSize;
  size_t aMaxGeneration;
  double aMutateRate;
  double aCrossoverRate;

};

}  // namespace ne
}  // namespace mlpack

#endif



