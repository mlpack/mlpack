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
  std::vector<Species> species;

  /** 
   * Default constructor.
   */
  Population():
    bestFitness(DBL_MAX),
    nextSpeciesId(0),
    nextGenomeId(0)
  {}

  /**
   * Parametric constructor.
   *
   * @param seedGenome Prototype of genomes in this population.
   * @param populationSize Number of total genomes in the population.
   */
  Population(Genome& seedGenome, int populationSize)
  {
    bestFitness = DBL_MAX;
    Species species(seedGenome, populationSize);
    this->species.push_back(species);  // NOTICE: we don't speciate.
    nextSpeciesId = 1;
    nextGenomeId = populationSize;
  }

  /**
   * Destructor.
   */
  ~Population() {}

  /**
   * Operator =.
   *
   * @param population The population to be compared with.
   */
  Population& operator =(const Population& population)
  {
    if (this != &population)
    {
      bestFitness = population.bestFitness;
      bestGenome = population.bestGenome;
      nextSpeciesId = population.nextSpeciesId;
      nextGenomeId = population.nextGenomeId;
      species = population.species;
    }

    return *this;
  }

  /**
   * Set best fitness.
   */
  void BestFitness(double bestFitness) { this->bestFitness = bestFitness; }

  /**
   * Get best fitness.
   */
  double BestFitness() const { return bestFitness; }

  /**
   * Set best genome.
   */
  void BestGenome(Genome& bestGenome) { this->bestGenome = bestGenome; }

  /**
   * Get best genome.
   */
  Genome BestGenome() const { return bestGenome; }

  /**
   * Set next species id.
   */
  void NextSpeciesId(int nextSpeciesId) { this->nextSpeciesId = nextSpeciesId; }

  /**
   * Get next species id.
   */
  int NextSpeciesId() const { return nextSpeciesId; }

  /**
   * Set next genome id.
   */
  void NextGenomeId(int nextGenomeId) { this->nextGenomeId = nextGenomeId; }

  /**
   * Get next genome id.
   */
  int NextGenomeId() const { return nextGenomeId; }

  /**
   * Get species number.
   */
  int NumSpecies() const { return species.size(); }

  /**
   * Get population size.
   */
  int PopulationSize() const
  {
    int populationSize = 0;
    for (int i = 0; i < species.size(); ++i)
    {
      populationSize += species[i].genomes.size();
    }

    return populationSize;
  }

  /**
   * Set best fitness to be the minimum of all genomes' fitness.
   */
  void SetBestFitnessAndGenome()
  {
    bestFitness = DBL_MAX;
    for (int i = 0; i < species.size(); ++i)
    {
      for (int j = 0; j < species[i].genomes.size(); ++j)
      {
        if (species[i].genomes[j].Fitness() < bestFitness)
        {
          bestFitness = species[i].genomes[j].Fitness();
          bestGenome = species[i].genomes[j];
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
    species.Id(nextSpeciesId);  // NOTICE: thus we changed species id when add to population.
    this->species.push_back(species);
    ++nextSpeciesId;
  }

  /**
   * Remove species.
   *
   * @param idx The index of species to remove in the population list.
   */
  void RemoveSpecies(int idx)
  {
    species.erase(species.begin() + idx);
  }

  /**
   * Re-assign genomes' id, from 0 to population size.
   */
  void ReassignGenomeId()
  {
    int id = -1;
    for (int i = 0; i < species.size(); ++i)
    {
      for (int j=0; j<species[i].genomes.size(); ++j)
      {
        ++id;
        species[i].genomes[j].Id(id);
      }
    }
    nextGenomeId = id + 1;
  }

 private:
  //! Best fitness.
  double bestFitness;

  //! Best genome.
  Genome bestGenome;

  //! Next species id.
  int nextSpeciesId;

  //! Next genome id.
  int nextGenomeId;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_POPULATION_HPP
