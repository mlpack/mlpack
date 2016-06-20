/**
 * @file population.hpp
 * @author Bang Liu
 *
 * Definition of Population class.
 */
#ifndef MLPACK_METHODS_NE_POPULATION_HPP
#define MLPACK_METHODS_NE_POPULATION_HPP

#include <cstddef>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"
#include "species.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a population consist of multiple species.
 */
class Population {
 public:
  // Species contained by this population.
  std::vector<Species> aSpecies;

  // Default constructor.

  // Parametric constructor.

  // Separates the population into species based on compatibility distance
  void Speciate();

  // Sort each species' genomes based on fitness.
  void Sort();

 
 private:
  // Number of species.
  size_t aNumSpecies;

  // Number of genomes including all species.
  size_t aPopulationSize;

  // Next species id.
  size_t aNextSpeciesId;

  // Next genome id.
  size_t aNextGenomeId;

  // Best fitness.
  double aBestFitness;

  // Best genome.
  Genome aBestGenome;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_POPULATION_HPP
