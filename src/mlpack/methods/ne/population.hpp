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

  // Genomes contained by this population.
  std::vector<Genome> aGenomes;

  // Default constructor.
  Population() {
    aNumSpecies = 0;
    aPopulationSize = 0;
    aBestFitness = DBL_MAX;
    aBestGenome = Genome();
    aNextSpeciesId = 0;
    aNextGenomeId = 0;
  }

  // Parametric constructor.
  Population(Genome& seedGenome, ssize_t populationSize) {
    aPopulationSize = populationSize;
    aNumSpecies = 0;
    aBestGenome = seedGenome;
    aBestFitness = DBL_MAX;
    Species species(seedGenome, populationSize);
    aGenomes = species.aGenomes;
    aNextSpeciesId = 0;
    aNextGenomeId = populationSize;
  }

  // Destructor.
  ~Population() {}

  // Operator =.
  Population& operator =(const Population& population) {
    if (this != &population) {
      aNumSpecies = population.aNumSpecies;
      aPopulationSize = population.aPopulationSize;
      aBestFitness = population.aBestFitness;
      aBestGenome = population.aBestGenome;
      aNextSpeciesId = population.aNextSpeciesId;
      aNextGenomeId = population.aNextGenomeId;
      aSpecies = population.aSpecies;
      aGenomes = population.aGenomes;
    }

    return *this;
  }

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

  // Set next genome id.
  void NextGenomeId(ssize_t nextGenomeId) { aNextGenomeId = nextGenomeId; }

  // Get next genome id.
  ssize_t NextGenomeId() const { return aNextGenomeId; }

  // Add species.
  void AddSpecies(Species& species) {
    aSpecies.push_back(species);
    aPopulationSize += species.SpeciesSize();
    ++aNumSpecies;  // TODO: do we really need numSpecies? Maybe NumSpecies() is enough.
    ++aNextSpeciesId;
  }

  // Remove species.
  void RemoveSpecies(ssize_t idx) {
    --aNumSpecies;
    aPopulationSize -= aSpecies[idx].aGenomes.size();
    aSpecies.erase(aSpecies.begin() + idx);
  }

  // ReassignGenomesId
  void ReassignGenomeId() {
    for (ssize_t i=0; i<aGenomes.size(); ++i) {
      aGenomes[i].Id(i);
    }
    aNextGenomeId = aGenomes.size();
  }

  // Aggregate genomes to aGenomes vector using species vector.
  void AggregateGenomes() {
    aGenomes.clear();
    for (ssize_t i=0; i<NumSpecies(); ++i) {
      for (ssize_t j=0; j<aSpecies[i].SpeciesSize(); ++j) {
        aGenomes.push_back(aSpecies[i].aGenomes[j]);
      }
    }
  }

  // Sort genomes by fitness. Smaller fitness is better and put first.
  void SortGenomes() {
    std::sort(aGenomes.begin(), aGenomes.end(), Species::CompareGenome);
  }

  // Get genome index in aGenomes.
  ssize_t GetGenomeIndex(ssize_t id) {
    for (ssize_t i=0; i<aPopulationSize; ++i) {
      if (aGenomes[i].Id() == id)
        return i;
    }
    return -1;
  }


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

  // Next genome id.
  ssize_t aNextGenomeId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_POPULATION_HPP
