 /**
 * @file neat.hpp
 * @author Bang Liu
 *
 * Definition of NEAT class.
 */
#ifndef MLPACK_METHODS_NE_NEAT_HPP
#define MLPACK_METHODS_NE_NEAT_HPP

#include <cstddef>
#include <cstdio>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"
#include "species.hpp"
#include "population.hpp"
#include "tasks.hpp"
#include "parameters.hpp"

namespace mlpack {
namespace ne {

struct LinkInnovation {
  ssize_t fromNeuronId;
  ssize_t toNeuronId;
  ssize_t newLinkInnovId;
};

struct NeuronInnovation {
  ssize_t splitLinkInnovId;
  ssize_t newNeuronId;
  ssize_t newInputLinkInnovId;
  ssize_t newOutputLinkInnovId;
};

/**
 * This class implements  NEAT algorithm.
 */
template<typename TaskType>
class NEAT {
 public:
  // Default constructor.

  // Parametric constructor.

  // Destructor.

  // Check whether a link innovation already exist.
  ssize_t CheckLinkInnovation(ssize_t fromNeuronId, ssize_t toNeuronId) {
    for (ssize_t i=0; i<aLinkInnovations.size(); ++i) {
      if (aLinkInnovations[i].fromNeuronId == fromNeuronId && 
          aLinkInnovations[i].toNeuronId == toNeuronId) {
        return i;
      }
    }
    
    return -1;  // -1 means no match found, a new innovation.
  }

  // Check whether a neuron innovation already exist.
  ssize_t CheckNeuronInnovation(ssize_t splitLinkInnovId) {
    for (ssize_t i=0; i<aNeuronInnovations.size(); ++i) {
      if (aNeuronInnovations[i].splitLinkInnovId == splitLinkInnovId) {
        return i;
      }
    }

    return -1;
  }

  // Add a new link innovation.
  LinkInnovation AddLinkInnovation(ssize_t fromNeuronId, ssize_t toNeuronId) {
    LinkInnovation linkInnov;
    linkInnov.fromNeuronId = fromNeuronId;
    linkInnov.toNeuronId = toNeuronId;
    linkInnov.newLinkInnovId = aNextLinkInnovId++;
    aLinkInnovations.push_back(linkInnov);

    return linkInnov;
  }

  // Add a new neuron innovation.
  NeuronInnovation AddNeuronInnovation(ssize_t splitLinkInnovId) {
    NeuronInnovation neuronInnov;
    neuronInnov.splitLinkInnovId = splitLinkInnovId;
    neuronInnov.newNeuronId = aNextNeuronId++;
    neuronInnov.newInputLinkInnovId = aNextLinkInnovId++;
    neuronInnov.newOutputLinkInnovId = aNextLinkInnovId++;
    aNeuronInnovations.push_back(neuronInnov);
    // TODO: do we need to add two link innovations ???

    return neuronInnov;
  }

  // Check if link exist or not.
  ssize_t IsLinkExist(Genome& genome, ssize_t fromNeuronId, ssize_t toNeuronId) {
    for (ssize_t i=0; i<genome.aLinkGenes.size(); ++i) {
      if (genome.aLinkGenes[i].FromNeuronId() == fromNeuronId &&
          genome.aLinkGenes[i].ToNeuronId() == toNeuronId) {
        return i;
      }
    }
    return -1;  // -1 means not exist.
  } 

  // Mutate: add new link to genome.
  void MutateAddLink(Genome& genome, double mutateAddLinkProb) {
    // Whether mutate or not.
    double p = mlpack::math::Random();
    if (p > mutateAddLinkProb) return;

    // Select from neuron
    ssize_t fromNeuronIdx = mlpack::math::RandInt(0, genome.aNeuronGenes.size());
    ssize_t fromNeuronId = genome.aNeuronGenes[fromNeuronIdx].Id();

    // Select to neuron which cannot be input.
    ssize_t toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.aNeuronGenes.size());
    ssize_t toNeuronId = genome.aNeuronGenes[toNeuronIdx].Id();

    // Check link already exist or not.
    ssize_t linkIdx = IsLinkExist(genome, fromNeuronId, toNeuronId);
    if (linkIdx != -1) {
      genome.aLinkGenes[linkIdx].Enabled(true);
      return;
    }

    // Check innovation already exist or not.
    ssize_t innovIdx = CheckLinkInnovation(fromNeuronId, toNeuronId);
    if (innovIdx != -1) {
      LinkGene linkGene(fromNeuronId,
                        toNeuronId,
                        aLinkInnovations[innovIdx].newLinkInnovId,
                        mlpack::math::RandNormal(0, 1), // TODO: make the distribution an argument for control?
                        true);
      genome.aLinkGenes.push_back(linkGene);
      return;
    }

    // If new link and new innovation, create it, push new innovation.
    LinkInnovation linkInnov = AddLinkInnovation(fromNeuronId, toNeuronId);
    LinkGene linkGene(fromNeuronId,
                      toNeuronId,
                      linkInnov.newLinkInnovId,
                      mlpack::math::RandNormal(0, 1), // TODO: make the distribution an argument for control?
                      true);
    genome.aLinkGenes.push_back(linkGene);
  }

  // Mutate: add new neuron to genome.
  void MutateAddNeuron(Genome& genome, double mutateAddNeuronProb) {
    // Whether mutate or not.
    double p = mlpack::math::Random();
    if (p > mutateAddNeuronProb) return;

    // No link.
    if (genome.aLinkGenes.size() == 0) return;

    // Select link to split.
    ssize_t linkIdx = mlpack::math::RandInt(0, genome.aLinkGenes.size());
    if (!genome.aLinkGenes[linkIdx].Enabled()) return;

    genome.aLinkGenes[linkIdx].Enabled(false);

    // Check innovation already exist or not.
    ssize_t splitLinkInnovId = genome.aLinkGenes[linkIdx].InnovationId();
    ssize_t innovIdx = CheckNeuronInnovation(splitLinkInnovId);
    if (innovIdx != -1) {
      NeuronGene neuronGene(aNeuronInnovations[innovIdx].newNeuronId,
                            HIDDEN,
                            SIGMOID,  // TODO: make it random??
                            0,
                            0);
      genome.aNeuronGenes.push_back(neuronGene);

      LinkGene inputLink(genome.aLinkGenes[linkIdx].FromNeuronId(),
                         aNeuronInnovations[innovIdx].newNeuronId,
                         aNeuronInnovations[innovIdx].newInputLinkInnovId,
                         1,
                         true);
      LinkGene outputLink(aNeuronInnovations[innovIdx].newNeuronId,
                          genome.aLinkGenes[linkIdx].ToNeuronId(),
                          aNeuronInnovations[innovIdx].newOutputLinkInnovId,
                          genome.aLinkGenes[linkIdx].Weight(),
                          true);
      genome.aLinkGenes.push_back(inputLink);
      genome.aLinkGenes.push_back(outputLink);
      return;
    }

    // If new innovation, create.
    NeuronInnovation neuronInnov = AddNeuronInnovation(splitLinkInnovId);

    NeuronGene neuronGene(neuronInnov.newNeuronId,
                          HIDDEN,
                          SIGMOID,  // TODO: make it random??
                          0,
                          0);
    genome.aNeuronGenes.push_back(neuronGene);

    LinkGene inputLink(genome.aLinkGenes[linkIdx].FromNeuronId(),
                       neuronInnov.newNeuronId,
                       neuronInnov.newInputLinkInnovId,
                       1,
                       true);
    LinkGene outputLink(neuronInnov.newNeuronId,
                        genome.aLinkGenes[linkIdx].ToNeuronId(),
                        neuronInnov.newOutputLinkInnovId,
                        genome.aLinkGenes[linkIdx].Weight(),
                        true);
    genome.aLinkGenes.push_back(inputLink);
    genome.aLinkGenes.push_back(outputLink);
  }

  // Crossover link weights.

  // Initialize population.

  // Reproduce next generation of population.

  // Evolve.
 
 private:
  std::vector<LinkInnovation> aLinkInnovations;
  std::vector<NeuronInnovation> aNeuronInnovations;
  ssize_t aNextNeuronId;
  ssize_t aNextLinkInnovId;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEAT_HPP
