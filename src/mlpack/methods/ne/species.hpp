/**
 * @file species.hpp
 * @author Bang Liu
 *
 * Definition of Species class.
 */
#ifndef MLPACK_METHODS_NE_SPECIES_HPP
#define MLPACK_METHODS_NE_SPECIES_HPP

#include <cstddef>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a species of genomes.
 */
class Species {
 public:
  // Genomes.
  std::vector<Genome> aGenomes;

  // Default constructor.
  Species() {
    aId = -1;
    aStaleAge = -1;
    aSpeciesSize = 0;
    aBestFitness = DBL_MAX;
    aBestGenome = Genome();
    aNextGenomeId = 0;
  }

  // Parametric constructor.
  // TODO: whether randomize, random range, as parameter or not??
  Species(Genome& seedGenome, ssize_t speciesSize) {
    aId = 0;
    aStaleAge = 0;
    aSpeciesSize = speciesSize;
    aBestFitness = DBL_MAX; // DBL_MAX denotes haven't evaluate yet.

    // Create genomes from seed Genome and randomize weight.
    for (ssize_t i=0; i<speciesSize; ++i) {
      Genome genome = seedGenome;
      genome.Id(i);
      aGenomes.push_back(genome);
      aGenomes[i].RandomizeWeights(-1, 1);
    }

    aNextGenomeId = speciesSize;
  }

  // Destructor.
  ~Species() {}

  // Operator =.
  Species& operator =(const Species& species) {
    if (this != &species) {
      aId = species.aId;
      aStaleAge = species.aStaleAge;
      aSpeciesSize = species.aSpeciesSize;
      aBestFitness = species.aBestFitness;
      aBestGenome = species.aBestGenome;
      aNextGenomeId = species.aNextGenomeId;
      aGenomes = species.aGenomes;
    }

    return *this;
  }

  // Set id.
  void Id(ssize_t id) { aId = id; }

  // Get id.
  ssize_t Id() const { return aId; }

  // Set age.
  void StaleAge(ssize_t staleAge) { aStaleAge = staleAge; }

  // Get age.
  ssize_t StaleAge() const { return aStaleAge; }

  // Set species size.
  void SpeciesSize(ssize_t speciesSize) { aSpeciesSize = speciesSize; }

  // Get species size.
  ssize_t SpeciesSize() const { return aSpeciesSize; }

  // Set best fitness.
  void BestFitness(double bestFitness) { aBestFitness = bestFitness; }

  // Get best fitness.
  double BestFitness() const { return aBestFitness; }

  // Set best fitness to be the minimum of all genomes' fitness.
  void SetBestFitness() {
    if (aGenomes.size() == 0) 
      return;

    aBestFitness = aGenomes[0].Fitness();
    for (ssize_t i=0; i<aGenomes.size(); ++i) {
      if (aGenomes[i].Fitness() < aBestFitness) {
        aBestFitness = aGenomes[i].Fitness();
      }
    }
  }

  // Sort genomes by fitness. First is best.
  static bool CompareGenome(Genome lg, Genome rg) {
    if (lg.Fitness() < rg.Fitness()) {  // NOTICE: we assume smaller is better.
      return true;
    } else if (rg.Fitness() < lg.Fitness()) {
      return false;
    } else if (lg.NumLink() < rg.NumLink()) {
      return true;
    } else if (rg.NumLink() < lg.NumLink()) {
      return false;
    } else if (mlpack::math::Random() < 0.5) {
      return true;
    } else {
      return false;
    }
  }
  void SortGenomes() {
    std::sort(aGenomes.begin(), aGenomes.end(), CompareGenome);
  }

  // Add new genome.
  void AddGenome(Genome& genome) {
    genome.Id(aNextGenomeId);  // NOTICE: thus we changed genome id when add to species.
    aGenomes.push_back(genome);
    ++aSpeciesSize;
    ++aNextGenomeId;
  }

 private:
  // Id of species.
  ssize_t aId;

  // Stale age (how many generations that its best fitness doesn't improve) of species.
  ssize_t aStaleAge;

  // Number of Genomes.
  ssize_t aSpeciesSize;

  // Best fitness.
  double aBestFitness;

  // Genome with best fitness.
  Genome aBestGenome;

  // Next genome id.
  ssize_t aNextGenomeId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_SPECIES_HPP
