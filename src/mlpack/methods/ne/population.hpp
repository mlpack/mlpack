/**
 * @file population.hpp
 * @author Bang Liu
 *
 * Definition of Population class.
 */
#ifndef MLPACK_METHODS_NE_POPULATION_HPP
#define MLPACK_METHODS_NE_POPULATION_HPP

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
class Population
{
 public:
  //! Species contained by this population.
  std::vector<Species> aSpecies;

  /** 
   * Default constructor.
   */
  Population():
    aBestFitness(DBL_MAX),
    aNextSpeciesId(0),
    aNextGenomeId(0)
  {}

  /**
   * Parametric constructor.
   *
   * @param seedGenome Prototype of genomes in this population.
   * @param populationSize Number of total genomes in the population.
   */
  Population(Genome& seedGenome, int populationSize)
  {
    aBestFitness = DBL_MAX;
    Species species(seedGenome, populationSize);
    aSpecies.push_back(species);  // NOTICE: we don't speciate.
    aNextSpeciesId = 1;
    aNextGenomeId = populationSize;
  }

  /**
   * Operator =.
   *
   * @param population The population to be compared with.
   */
  Population& operator =(const Population& population)
  {
    if (this != &population)
    {
      aBestFitness = population.aBestFitness;
      aBestGenome = population.aBestGenome;
      aNextSpeciesId = population.aNextSpeciesId;
      aNextGenomeId = population.aNextGenomeId;
      aSpecies = population.aSpecies;
    }

    return *this;
  }

  /**
   * Set best fitness.
   */
  void BestFitness(double bestFitness) { aBestFitness = bestFitness; }

  /**
   * Get best fitness.
   */
  double BestFitness() const { return aBestFitness; }

  /**
   * Set best genome.
   */
  void BestGenome(Genome& bestGenome) { aBestGenome = bestGenome; }

  /**
   * Get best genome.
   */
  Genome BestGenome() const { return aBestGenome; }

  /**
   * Set next species id.
   */
  void NextSpeciesId(int nextSpeciesId) { aNextSpeciesId = nextSpeciesId; }

  /**
   * Get next species id.
   */
  int NextSpeciesId() const { return aNextSpeciesId; }

  /**
   * Set next genome id.
   */
  void NextGenomeId(int nextGenomeId) { aNextGenomeId = nextGenomeId; }

  /**
   * Get next genome id.
   */
  int NextGenomeId() const { return aNextGenomeId; }

  /**
   * Get species number.
   */
  int NumSpecies() const { return aSpecies.size(); }

  /**
   * Get population size.
   */
  int PopulationSize() const
  {
    int populationSize = 0;
    for (int i=0; i<aSpecies.size(); ++i)
    {
      populationSize += aSpecies[i].aGenomes.size();
    }

    return populationSize;
  }

  /**
   * Set best fitness to be the minimum of all genomes' fitness.
   */
  void SetBestFitnessAndGenome()
  {
    aBestFitness = DBL_MAX;
    for (int i=0; i<aSpecies.size(); ++i)
    {
      for (int j=0; j<aSpecies[i].aGenomes.size(); ++j)
      {
        if (aSpecies[i].aGenomes[j].Fitness() < aBestFitness)
        {
          aBestFitness = aSpecies[i].aGenomes[j].Fitness();
          aBestGenome = aSpecies[i].aGenomes[j];
        }
      }
    }
  }

  /**
   * Add species.
   *
   * @param species The species to add.
   */
  void AddSpecies(Species& species)
  {
    species.Id(aNextSpeciesId);  // NOTICE: thus we changed species id when add to population.
    aSpecies.push_back(species);
    ++aNextSpeciesId;
  }

  /**
   * Remove species.
   *
   * @param idx The index of species to remove in the population list.
   */
  void RemoveSpecies(int idx)
  {
    aSpecies.erase(aSpecies.begin() + idx);
  }

  /**
   * Re-assign genomes' id, from 0 to population size.
   */
  void ReassignGenomeId()
  {
    int id = -1;
    for (int i=0; i<aSpecies.size(); ++i)
    {
      for (int j=0; j<aSpecies[i].aGenomes.size(); ++j)
      {
        ++id;
        aSpecies[i].aGenomes[j].Id(id);
      }
    }
    aNextGenomeId = id + 1;
  }

 private:
  //! Best fitness.
  double aBestFitness;

  //! Best genome.
  Genome aBestGenome;

  //! Next species id.
  int aNextSpeciesId;

  //! Next genome id.
  int aNextGenomeId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_POPULATION_HPP
