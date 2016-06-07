/**
 * @file population.hpp
 * @author Bang Liu
 *
 * Definition of Population class.
 */
#ifndef MLPACK_METHODS_NE_POPULATION_HPP
#define MLPACK_METHODS_NE_POPULATION_HPP

#include <mlpack/core.hpp>

#include "gene.hpp"
#include "genome.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a population of genomes.
 */
class Population {
 public:
  // Default constructor.
  Population() {}

  // Parametric constructor.

  // Copy constructor.

  // Destructor.
  ~Population() {}

 private:
  // Genomes.
  std::vector<Genome> aGenomes;

  // Number of Genomes.
  unsigned int aNumGenome;

  // Best fitness.
  double aBestFitness;

  // Genome with best fitness.
  Genome aBestGenome;

  // Next genome id.
  unsigned int aNextGenomeId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_POPULATION_HPP

