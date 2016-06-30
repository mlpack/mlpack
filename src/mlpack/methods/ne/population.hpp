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
    aNextSpeciesId = 0;
    aNextGenomeId = 0;
  }

  // Parametric constructor.
  Population(Genome& seedGenome, ssize_t populationSize) {
    aPopulationSize = populationSize;
    aNumSpecies = 1;
    aBestFitness = DBL_MAX;
    Species species(seedGenome, populationSize);
    aSpecies.push_back(species);  // NOTICE: we don't speciate.
    aNextSpeciesId = 1;
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

  // Set best fitness to be the minimum of all genomes' fitness.
  void SetBestFitness() {
    aBestFitness = DBL_MAX;
    for (ssize_t i=0; i<aSpecies.size(); ++i) {
      for (ssize_t j=0; j<aSpecies[i].aGenomes.size(); ++j) {
        if (aSpecies[i].aGenomes[j].Fitness() < aBestFitness) {
          aBestFitness = aSpecies[i].aGenomes[j].Fitness();
        }
      }
    }
  }

  // Add species.
  void AddSpecies(Species& species) {
    aSpecies.push_back(species);
    aPopulationSize += species.aGenomes.size();
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
    ssize_t id = -1;
    for (ssize_t i=0; i<aSpecies.size(); ++i) {
      for (ssize_t j=0; j<aSpecies[i].aGenomes.size(); ++j) {
        ++id;
        aSpecies[i].aGenomes[j].Id(id);
      }
    }
    aNextGenomeId = id + 1;
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
