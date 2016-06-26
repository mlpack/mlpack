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
    for (ssize_t i=0; i<genome.NumLink(); ++i) {
      if (genome.aLinkGenes[i].FromNeuronId() == fromNeuronId &&
          genome.aLinkGenes[i].ToNeuronId() == toNeuronId) {
        return i;
      }
    }
    return -1;  // -1 means not exist.
  } 

  // Mutate: add new link to genome.
  // TODO: what if created looped link? It will influence the depth calculation in genome class!!
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
      genome.AddLink(linkGene);
      return;
    }

    // If new link and new innovation, create it, push new innovation.
    LinkInnovation linkInnov = AddLinkInnovation(fromNeuronId, toNeuronId);
    LinkGene linkGene(fromNeuronId,
                      toNeuronId,
                      linkInnov.newLinkInnovId,
                      mlpack::math::RandNormal(0, 1), // TODO: make the distribution an argument for control?
                      true);
    genome.AddLink(linkGene);
  }

  // Mutate: add new neuron to genome.
  void MutateAddNeuron(Genome& genome, double mutateAddNeuronProb) {
    // Whether mutate or not.
    double p = mlpack::math::Random();
    if (p > mutateAddNeuronProb) return;

    // No link.
    if (genome.NumLink() == 0) return;

    // Select link to split.
    ssize_t linkIdx = mlpack::math::RandInt(0, genome.NumLink());
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
      genome.AddHiddenNeuron(neuronGene);

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
      genome.AddLink(inputLink);
      genome.AddLink(outputLink);
      return;
    }

    // If new innovation, create.
    NeuronInnovation neuronInnov = AddNeuronInnovation(splitLinkInnovId);

    NeuronGene neuronGene(neuronInnov.newNeuronId,
                          HIDDEN,
                          SIGMOID,  // TODO: make it random??
                          0,
                          0);
    genome.AddHiddenNeuron(neuronGene);

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
    genome.AddLink(inputLink);
    genome.AddLink(outputLink);
  }

  // Crossover link weights.
  // NOTICE: assume momGenome is the better genome.
  // NOTICE: assume childGenome is empty.
  void CrossoverLinkAndNeuron(Genome& momGenome, Genome& dadGenome, Genome& childGenome) {
    // Add input and output neuron genes to child genome.
    for (ssize_t i=0; i<(momGenome.NumInput() + momGenome.NumOutput()); ++i) {
      childGenome.aNeuronGenes.push_back(momGenome.aNeuronGenes[i]);
    }

    // Iterate to add link genes and neuron genes to child genome.
    for (ssize_t i=0; i<momGenome.NumLink(); ++i) {
      ssize_t innovId = momGenome.aLinkGenes[i].InnovationId();
      ssize_t idx = dadGenome.GetLinkIndex(innovId);

      if (idx == -1 && momGenome.aLinkGenes[i].Enabled()) {  // exceed or disjoint
        childGenome.AddLink(momGenome.aLinkGenes[i]);
        
        // Add from neuron
        ssize_t idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        ssize_t idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron
        idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        continue;
      }

      if (idx != -1 && momGenome.aLinkGenes[i].Enabled() && mlpack::math::Random() < 0.5) {
        childGenome.AddLink(momGenome.aLinkGenes[i]);

        // Add from neuron
        ssize_t idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        ssize_t idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron
        idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        continue;
      }

      if (idx != -1 && momGenome.aLinkGenes[i].Enabled() && mlpack::math::Random() >= 0.5) {
        childGenome.AddLink(dadGenome.aLinkGenes[idx]);

        // Add from neuron   TODO: make it a function?? check whether crossover is correct.
        ssize_t idxInChild = childGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].FromNeuronId());
        ssize_t idxInParent = dadGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].FromNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(dadGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron
        idxInChild = childGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].ToNeuronId());
        idxInParent = dadGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].ToNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(dadGenome.aNeuronGenes[idxInParent]);
        }

        continue;
      }
    }
  }

  void Crossover(Genome& genome1, Genome& genome2, Genome& childGenome) {
    if (Species::CompareGenome(genome1, genome2)) {  // genome1 is better
      CrossoverLinkAndNeuron(genome1, genome2, childGenome);
    } else {
      CrossoverLinkAndNeuron(genome2, genome1, childGenome);
    }
  }

  // Measure two genomes' disjoint (including exceed).
  // TODO: we can seperate into disjoint and exceed.
  // But currently maybe it is enough.
  double Disjoint(Genome& genome1, Genome& genome2) {
    double numDisjoint = 0;

    for (ssize_t i=0; i<genome1.NumLink(); ++i) {
      ssize_t innovId = genome1.aLinkGenes[i].InnovationId();
      ssize_t idx = genome2.GetLinkIndex(innovId);
      if (idx == -1 && genome1.aLinkGenes[i].Enabled()) {
        ++numDisjoint;
      }
    }

    for (ssize_t i=0; i<genome2.NumLink(); ++i) {
      ssize_t innovId = genome2.aLinkGenes[i].InnovationId();
      ssize_t idx = genome1.GetLinkIndex(innovId);
      if (idx == -1 && genome2.aLinkGenes[i].Enabled()) {
        ++numDisjoint;
      }
    }

    ssize_t largerGenomeSize = std::max(genome1.NumLink(), genome2.NumLink());
    double deltaD = numDisjoint / largerGenomeSize;
    return deltaD; 
  }

  // Measure two genomes' weight difference.
  double WeightDiff(Genome& genome1, Genome& genome2) {
    double deltaW = 0;
    ssize_t coincident = 0;

    for (ssize_t i=0; i<genome1.NumLink(); ++i) {
      ssize_t innovId = genome1.aLinkGenes[i].InnovationId();
      ssize_t idx = genome2.GetLinkIndex(innovId);
      if (idx != -1 && genome1.aLinkGenes[i].Enabled()) {
        deltaW += std::abs(genome1.aLinkGenes[i].Weight() - genome2.aLinkGenes[idx].Weight());
        ++coincident;
      }
    }

    deltaW = deltaW / coincident;
    return deltaW;
  }

  // Whether two genome belong to same species or not.
  bool IsSameSpecies(Genome& genome1, Genome& genome2) {
    double deltaD = Disjoint(genome1, genome2);
    double deltaW = WeightDiff(genome1, genome2);
    double delta = aCoeffDisjoint * deltaD + aCoeffWeightDiff * deltaW;

    if (delta < aCompatThreshold) {
      return true;
    } else {
      return false;
    }
  }

  // Add genome to existing species or create new species.
  void AddGenomeToSpecies(Genome& genome) {
    for (ssize_t i=0; i<aPopulation.NumSpecies(); ++i) {
      if (aPopulation.aSpecies[i].SpeciesSize() > 0) {
        if (IsSameSpecies(aPopulation.aSpecies[i].aGenomes[0], genome)) {  // each first genome in species is the representative genome.
          aPopulation.aSpecies[i].AddGenome(genome);
          return;
        }
      }
    }

    Species newSpecies = Species();
    newSpecies.AddGenome(genome);
    newSpecies.Id(aPopulation.NextSpeciesId());  // NOTICE: changed species id.
    aPopulation.AddSpecies(newSpecies);
  }

  // Set adjusted fitness.
  void AdjustFitness() {
    for (ssize_t i=0; i<aPopulation.NumSpecies(); ++i) {
      if (aPopulation.aSpecies[i].SpeciesSize() > 0) {
        for (ssize_t j=0; j<aPopulation.aSpecies[i].SpeciesSize(); ++j) {
          double fitness = aPopulation.aSpecies[i].aGenomes[j].Fitness();
          ssize_t speciesSize = aPopulation.aSpecies[i].SpeciesSize();
          double adjustedFitness = fitness / speciesSize;
          aPopulation.aSpecies[i].aGenomes[j].AdjustedFitness(adjustedFitness);
        }
      }
    }
  }

  // Distribute genomes into species.
  void Speciate() {

  }

  // Initialize population.
  void InitPopulation() {
    aPopulation = Population(aSeedGenome, aPopulationSize);
  }

  // Reproduce next generation of population.
  void Reproduce() {

  }

  // Evolve.
 
 private:
  // Task.
  TaskType aTask;

  // Seed genome. It is used for init species.
  Genome aSeedGenome;

  // Population to evolve.
  Population aPopulation;

  // Population size.
  ssize_t aPopulationSize;

  // List of link innovations.
  std::vector<LinkInnovation> aLinkInnovations;

  // List of neuron innovations.
  std::vector<NeuronInnovation> aNeuronInnovations;

  // Next neuron id.
  ssize_t aNextNeuronId;

  // Next link id.
  ssize_t aNextLinkInnovId;

  // Max number of generation to evolve.
  ssize_t aMaxGeneration;

  // Efficient for disjoint.
  double aCoeffDisjoint;

  // Efficient for weight difference.
  double aCoeffWeightDiff;

  // Threshold for judge whether belong to same species.
  double aCompatThreshold;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEAT_HPP
