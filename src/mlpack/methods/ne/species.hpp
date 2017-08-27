/**
 * @file species.hpp
 * @author Bang Liu
 *
 * Definition of Species class.
 */
#ifndef MLPACK_METHODS_NE_SPECIES_HPP
#define MLPACK_METHODS_NE_SPECIES_HPP

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a species of genomes.
 */
class Species
{
 public:
  //! Genomes.
  std::vector<Genome> genomes;

  /**
   * Default constructor.
   */
  Species():
    id(-1),
    staleAge(-1),
    bestFitness(DBL_MAX),  // DBL_MAX denotes haven't evaluate yet.
    nextGenomeId(0),
    previousFitness(DBL_MAX),
    stagnation(0)
  {}

  /**
   * Parametric constructor.
   *
   * @param seedGenome This genome is the prototype of all genomes in the species.
   *                   Each genome will have same structure with seedGenome, and 
   *                   then randomize weights.
   * @param speciesSize Number of genomes in this species.
   */
  Species(Genome& seedGenome, int speciesSize)
  {
    id = 0;
    staleAge = 0;
    bestFitness = DBL_MAX; 
    nextGenomeId = speciesSize;
    previousFitness = DBL_MAX;
    stagnation = 0;

    // Create genomes from seed Genome and randomize weight.
    for (int i = 0; i < speciesSize; ++i)
    {
      Genome genome = seedGenome;
      genome.Id(i);
      genomes.push_back(genome);
      genomes[i].RandomizeWeights(-1, 1);
    }
  }

  // *
  //  * Destructor.
   
  // ~Species() {}

  // /**
  //  * Operator =.
  //  *
  //  * @param species Compare with this species.
  //  */
  // Species& operator =(const Species& species)
  // {
  //   if (this != &species)
  //   {
  //     id = species.id;
  //     staleAge = species.staleAge;
  //     bestFitness = species.bestFitness;
  //     bestGenome = species.bestGenome;
  //     nextGenomeId = species.nextGenomeId;
  //     genomes = species.genomes;
  //     stagnation = species.stagnation;
  //   }

  //   return *this;
  // }

  /**
   * Set id.
   */
  void Id(int id) { this->id = id; }

  /**
   * Get id.
   */
  int Id() const { return id; }

  /**
   * Set age.
   */
  void StaleAge(int staleAge) { this->staleAge = staleAge; }

  /**
   * Get age.
   */
  int StaleAge() const { return staleAge; }

  /**
   * Set best fitness.
   */
  void BestFitness(double bestFitness) { this->bestFitness = bestFitness; }

  /**
   * Get best fitness.
   */
  double BestFitness() const { return bestFitness; }

  /**
   * Get species size.
   */
  int SpeciesSize() const { return genomes.size(); }


  double& Fitness() { return fitness; }

  double Fitness() const { return fitness; }

  size_t Stagnation() const { return stagnation; }
  size_t& Stagnation() { 

    // std::cout << "id: " << id << std::endl;
    return stagnation; }

  double PreviousFitness() const { return previousFitness; }
  double& PreviousFitness() { return previousFitness; }


  size_t Size() const { return genomes.size(); }





  /**
   * Set best fitness to be the minimum of all genomes' fitness.
   */
  void SetBestFitnessAndGenome()
  {
    if (genomes.size() == 0) 
      return;

    bestFitness = genomes[0].Fitness();
    for (int i = 0; i < genomes.size(); ++i)
    {
      if (genomes[i].Fitness() < bestFitness)
      {
        bestFitness = genomes[i].Fitness();
        bestGenome = genomes[i];
      }
    }
  }

  /**
   * Sort genomes by fitness. First is best.
   */
  static bool CompareGenome(const Genome& lg, const Genome& rg)
  {
    return (lg.Fitness() < rg.Fitness());
  }
  void SortGenomes()
  {
    std::sort(genomes.begin(), genomes.end(), CompareGenome);
  }

  /**
   * Add new genome.
   *
   * @param genome The genome to add.
   */
  void AddGenome(Genome& genome)
  {
    genome.Id(nextGenomeId);  // NOTICE: thus we changed genome id when add to species.
    genomes.push_back(genome);
    ++nextGenomeId;
  }

  void Add(Genome& genome)
  {
    genome.Id(nextGenomeId);  // NOTICE: thus we changed genome id when add to species.
    genomes.push_back(genome);
    ++nextGenomeId;
  }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    std::cout << "SPECIES load\n";
    ar & data::CreateNVP(id, "id");
    ar & data::CreateNVP(staleAge, "staleAge");
    ar & data::CreateNVP(bestFitness, "bestFitness");
    ar & data::CreateNVP(bestGenome, "bestGenome");
    ar & data::CreateNVP(nextGenomeId, "nextGenomeId");
    ar & data::CreateNVP(genomes, "genomes");
  }

/**
 * Non-intrusive serialization for Neighbor Search class. We need this
 * definition because we are going to use the serialize function for boost
 * variant, which will look for a serialize function for its member types.
 */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    Serialize(ar, version);
  }

 private:
  //! Id of species.
  int id;

  //! Stale age (how many generations that its best fitness doesn't improve) of species.
  int staleAge;

  //! Best fitness.
  double bestFitness;

  //! Genome with best fitness.
  Genome bestGenome;

  //! Next genome id.
  int nextGenomeId;

  double fitness;
  double previousFitness;

  size_t stagnation;

};

}  // namespace ne
}  // namespace mlpack

# endif  // MLPACK_METHODS_NE_SPECIES_HPP
