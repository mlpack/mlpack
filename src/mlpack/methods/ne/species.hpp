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
    aSpeciesSize = 0;
    aBestFitness = DBL_MAX;
    aBestGenome = Genome();
    aNextGenomeId = 0;
  }

  // Parametric constructor.
  // TODO: whether randomize, random range, as parameter or not??
  Species(Genome& seedGenome, size_t speciesSize) {
    aSpeciesSize = speciesSize;
    aBestFitness = DBL_MAX; // DBL_MAX denotes haven't evaluate yet.

    // Create genomes from seed Genome and randomize weight.
    for (size_t i=0; i<speciesSize; ++i) {
      Genome genome = seedGenome;
      genome.Id(i);
      aGenomes.push_back(genome);
      aGenomes[i].RandomizeWeights(-1, 1);
    }

    aNextGenomeId = speciesSize;
  }

  size_t SpeciesSize() { return aSpeciesSize; }

  // Destructor.
  ~Species() {}

  // Set/get best fitness.
  double& BestFitness() { return aBestFitness; }

  // Set best fitness to be the minimum of all genomes' fitness.
  void SetBestFitness() {
    if (aGenomes.size() == 0) 
      return;

    aBestFitness = aGenomes[0].Fitness();
    for (size_t i=0; i<aGenomes.size(); ++i) {
      if (aGenomes[i].Fitness() < aBestFitness) {
        aBestFitness = aGenomes[i].Fitness();
      }
    }
  }

  // Sort genomes by fitness. First is best.
  static bool CompareGenome(Genome lg, Genome rg) {
    return (lg.Fitness() < rg.Fitness());
  }
  bool SortSpecies() {
    std::sort(aGenomes.begin(), aGenomes.end(), CompareGenome);
  }

 private:

  // Number of Genomes.
  size_t aSpeciesSize;

  // Best fitness.
  double aBestFitness;

  // Genome with best fitness.
  Genome aBestGenome;

  // Next genome id.
  size_t aNextGenomeId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_SPECIES_HPP

