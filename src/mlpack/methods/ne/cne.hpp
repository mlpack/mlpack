 /**
 * @file cne.hpp
 * @author Bang Liu
 *
 * Definition of CNE class.
 */
#ifndef MLPACK_METHODS_NE_POPULATION_HPP
#define MLPACK_METHODS_NE_POPULATION_HPP

#include <mlpack/core.hpp>

#include "gene.hpp"
#include "genome.hpp"
#include "population.hpp"

namespace mlpack {
namespace ne {

/**
 * This class implements Conventional Neuro-evolution (CNE): weight
 * evolution on topologically fixed neural networks.
 */
class CNE {
 public:
  // Default constructor.

  // Parametric constructor.

  // Destructor.

  // Encode neural network to genome.
  void BuildPhenotype(Genome& genome, FFN& ffn) const {

  }

  // Decode genome to neural network.
  void BuildGenotype(FFN& ffn, Genome& genome) const {

  }

  // Initializing the population of genomes.

  // Mutate operator.

  // Crossover operator.

  // Reproduce.

  // Evolution of population.
  bool Evolution() {
    // Generate initial population G(0) at random
    unsigned int i = 0;
    
    // Repeat
    while (i < maxIter) {
    	// Decode each genotype in population into ANN and evaluate it.
    	// If termination criterion is satisfied, return.

    	// Select parents from G(i) based on their fitness.

    	// Apply search operators to parents and produce offspring which
    	// form G(i + 1).

    }
  }

 private:

};

}  // namespace ne
}  // namespace mlpack

#endif



