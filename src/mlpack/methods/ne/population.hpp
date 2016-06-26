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
  Population() {
    aNumSpecies = 0;
    aPopulationSize = 0;
    aBestFitness = DBL_MAX;
    aBestGenome = Genome();
    aNextSpeciesId = 0;
  }

  // Parametric constructor.
  Population(Genome& seedGenome, ssize_t populationSize) {
    aPopulationSize = populationSize;
    aNumSpecies = 1;
    aBestGenome = seedGenome;
    aBestFitness = DBL_MAX;
    Species species(seedGenome, populationSize);
    species.Id(0);
    aSpecies.push_back(species);
    aNextSpeciesId = 1;
  }

  // Destructor.
  ~Population() {}

  // Set species number.
  void NumSpecies(ssize_t numSpecies) { aNumSpecies = numSpecies; }

  // Get species number.
  ssize_t NumSpecies() const { return aNumSpecies; }

  // Set population size.
  void PopulationSize(ssize_t populationSize) { aPopulationSize = populationSize; }

  // Get population size.
  ssize_t PopulationSize() const { return aPopulationSize; }

  // Set best fitness.
  void BestFitness(double bestFitness) { aBestFitness = bestFitness; }

  // Get best fitness.
  double BestFitness() const { return aBestFitness; }

  // Set best genome.
  void BestGenome(Genome& bestGenome) { aBestGenome = bestGenome; }

  // Get best genome.
  Genome BestGenome() const { return aBestGenome; }

  // Set next species id.
  void NextSpeciesId(ssize_t nextSpeciesId) { aNextSpeciesId = nextSpeciesId; }

  // Get next species id.
  ssize_t NextSpeciesId() const { return aNextSpeciesId; }

  // Add species.
  void AddSpecies(Species& species) {
    aSpecies.push_back(species);
    aPopulationSize += species.SpeciesSize();
    ++aNumSpecies;
    ++aNextSpeciesId;
  }

  // Separates the population into species based on compatibility distance
  void Speciate();

  // Sort each species' genomes based on fitness.
  void Sort();

 private:
  // Number of species.
  ssize_t aNumSpecies;

  // Number of genomes including all species.
  ssize_t aPopulationSize;

  // Best fitness.
  double aBestFitness;

  // Best genome.
  Genome aBestGenome;

  // Next species id.
  ssize_t aNextSpeciesId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_POPULATION_HPP
