/**
 * @file genome.hpp
 * @author Bang Liu
 *
 * Definition of the Genome class.
 */
#ifndef MLPACK_METHODS_NE_GENOME_HPP
#define MLPACK_METHODS_NE_GENOME_HPP

#include <mlpack/core.hpp>

#include "gene.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a genome.
 # A genome is consist of a group of genes.
 */
class Genome {
 public:
  // Default constructor.
  Genome() {}
  
  // Parametric constructor.
  Genome(unsigned int id,
  	     std::vector<NeuronGene> neuronGenes,
         std::vector<LinkGene> linkGenes,
         unsigned int depth,
         double fitness):
    aId(id),
    aNeuronGenes(neuronGenes),
    aLinkGenes(linkGenes),
    aDepth(depth),
    aFitness(fitness)
  {}

  // Copy constructor.
  Genome(const Genome& genome) {
    aId = genome.aId;
    aNeuronGenes = genome.aNeuronGenes;
    aLinkGenes = genome.aLinkGenes;
    aDepth = genome.aDepth;
    aFitness = genome.aFitness;
  }

  // Destructor.
  ~Genome() {}

  // Get genome id.
  unsigned int Id() const { return aId; }

  // Get depth.
  unsigned int Depth() const { return aDepth; }

  // Set depth.
  void Depth(unsigned int depth) { aDepth = depth; }

  // Get fitness.
  double Fitness() const { return aFitness; }

  // Set fitness.
  void Fitness(double fitness) { aFitness = fitness; }

  // Operator =.
  Genome& operator =(const Genome& genome) {
    if (this != &genome) {
      aId = genome.aId;
      aNeuronGenes = genome.aNeuronGenes;
      aLinkGenes = genome.aLinkGenes;
      aDepth = genome.aDepth;
      aFitness = genome.aFitness;
    }

    return *this;
  }

 private:
  // Genome id.
  unsigned int aId;

  // Neurons.
  std::vector<NeuronGene> aNeuronGenes;

  // Links.
  std::vector<LinkGene> aLinkGenes;

  // Network maximum depth.
  unsigned int aDepth;

  // Genome fitness.
  double aFitness;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENOME_HPP