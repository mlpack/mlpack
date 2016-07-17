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
#include <numeric>

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
  // Parametric constructor.
  NEAT(TaskType task, Genome& seedGenome, Parameters& params) {
    aTask = task;
    aSeedGenome = seedGenome;
    aNextNeuronId = seedGenome.NumNeuron();
    aNextLinkInnovId = seedGenome.NumLink();
    aPopulationSize = params.aPopulationSize;
    aMaxGeneration = params.aMaxGeneration;
    aCoeffDisjoint = params.aCoeffDisjoint;
    aCoeffWeightDiff = params.aCoeffWeightDiff;
    aCompatThreshold = params.aCompatThreshold;
    aStaleAgeThreshold = params.aStaleAgeThreshold;
    aCrossoverRate = params.aCrossoverRate;
    aCullSpeciesPercentage = params.aCullSpeciesPercentage;
    aMutateWeightProb = params.aMutateWeightProb;
    aPerturbWeightProb = params.aPerturbWeightProb;
    aMutateWeightSize = params.aMutateWeightSize;
    aMutateAddLinkProb = params.aMutateAddLinkProb;
    aMutateAddRecurrentLinkProb = params.aMutateAddRecurrentLinkProb;
    aMutateAddLoopLinkProb = params.aMutateAddLoopLinkProb;
    aMutateAddNeuronProb = params.aMutateAddNeuronProb;
    aMutateEnabledProb = params.aMutateEnabledProb;
    aMutateDisabledProb = params.aMutateDisabledProb;
  }

  // Destructor.
  ~NEAT() {}

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
  void AddLinkInnovation(ssize_t fromNeuronId, ssize_t toNeuronId, LinkInnovation& linkInnov) {
    linkInnov.fromNeuronId = fromNeuronId;
    linkInnov.toNeuronId = toNeuronId;
    linkInnov.newLinkInnovId = aNextLinkInnovId++;
    aLinkInnovations.push_back(linkInnov);
  }

  // Add a new neuron innovation.
  void AddNeuronInnovation(ssize_t splitLinkInnovId, NeuronInnovation& neuronInnov) {
    neuronInnov.splitLinkInnovId = splitLinkInnovId;
    neuronInnov.newNeuronId = aNextNeuronId++;
    neuronInnov.newInputLinkInnovId = aNextLinkInnovId++;
    neuronInnov.newOutputLinkInnovId = aNextLinkInnovId++;
    aNeuronInnovations.push_back(neuronInnov);
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
  void MutateAddLink(Genome& genome,
                     double mutateAddLinkProb,
                     double mutateAddRecurrentLinkProb,
                     double mutateAddLoopLinkProb) {
    // Whether mutate or not.
    double p = mlpack::math::Random();
    if (p > mutateAddLinkProb) return;

    if (genome.aNeuronGenes.size() == 0) return;

    // Select from neuron
    ssize_t fromNeuronIdx = mlpack::math::RandInt(0, genome.aNeuronGenes.size());
    ssize_t fromNeuronId = genome.aNeuronGenes[fromNeuronIdx].Id();

    // Select to neuron which cannot be input.
    ssize_t toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.aNeuronGenes.size()); 
    ssize_t toNeuronId = genome.aNeuronGenes[toNeuronIdx].Id();

    // Handle same depth connection, recurrent connection and loop connection.
    if (genome.aNeuronGenes[fromNeuronIdx].Depth() == genome.aNeuronGenes[toNeuronIdx].Depth()) {
      return;
    } else if (fromNeuronId == toNeuronId) {
      p = mlpack::math::Random();
      if (p >= mutateAddRecurrentLinkProb) return;
    } else if (genome.aNeuronGenes[fromNeuronIdx].Depth() > genome.aNeuronGenes[toNeuronIdx].Depth()) {
      p = mlpack::math::Random();
      if (p >= mutateAddLoopLinkProb) return;
    }

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
                        mlpack::math::RandNormal(0, 1),
                        true);
      genome.AddLink(linkGene);
      return;
    }

    // If new link and new innovation, create it, push new innovation.
    LinkInnovation linkInnov;
    AddLinkInnovation(fromNeuronId, toNeuronId, linkInnov);
    LinkGene linkGene(fromNeuronId,
                      toNeuronId,
                      linkInnov.newLinkInnovId,
                      mlpack::math::RandNormal(0, 1),
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
    NeuronGene fromNeuron;
    genome.GetNeuronById(genome.aLinkGenes[linkIdx].FromNeuronId(), fromNeuron);
    NeuronGene toNeuron;
    genome.GetNeuronById(genome.aLinkGenes[linkIdx].ToNeuronId(), toNeuron);

    // Check innovation already exist or not.
    ssize_t splitLinkInnovId = genome.aLinkGenes[linkIdx].InnovationId();
    ssize_t innovIdx = CheckNeuronInnovation(splitLinkInnovId);

    // If existing innovation.
    if (innovIdx != -1) {
      NeuronGene neuronGene(aNeuronInnovations[innovIdx].newNeuronId,
                            HIDDEN,
                            SIGMOID,  // TODO: make it random??
                            (fromNeuron.Depth() + toNeuron.Depth()) / 2,
                            0,
                            0);
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
      genome.AddHiddenNeuron(neuronGene);
      genome.AddLink(inputLink);
      genome.AddLink(outputLink);
      return;
    }

    // If new innovation, create.
    // Add neuron innovation, input link innovation, output innovation.
    NeuronInnovation neuronInnov;
    AddNeuronInnovation(splitLinkInnovId, neuronInnov);

    LinkInnovation inputLinkInnov;
    inputLinkInnov.fromNeuronId = genome.aLinkGenes[linkIdx].FromNeuronId();
    inputLinkInnov.toNeuronId = neuronInnov.newNeuronId;
    inputLinkInnov.newLinkInnovId = neuronInnov.newInputLinkInnovId;
    aLinkInnovations.push_back(inputLinkInnov);

    LinkInnovation outputLinkInnov;
    outputLinkInnov.fromNeuronId = neuronInnov.newNeuronId;
    outputLinkInnov.toNeuronId = genome.aLinkGenes[linkIdx].ToNeuronId();
    outputLinkInnov.newLinkInnovId = neuronInnov.newOutputLinkInnovId;
    aLinkInnovations.push_back(outputLinkInnov);
    
    // Add neuron, input link, output link.
    NeuronGene neuronGene(neuronInnov.newNeuronId,
                          HIDDEN,
                          SIGMOID,  // TODO: make it random??
                          (fromNeuron.Depth() + toNeuron.Depth()) / 2,
                          0,
                          0);
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
    genome.AddHiddenNeuron(neuronGene);
    genome.AddLink(inputLink);
    genome.AddLink(outputLink);
  }

  // Mutate: enable disabled, or disable enabled link.
  void MutateEnableDisable(Genome& genome, bool enabled, double mutateProb) {
    double p = mlpack::math::Random();
    if (p > mutateProb) return;

    std::vector<ssize_t> linkIndexs;
    for (ssize_t i=0; i<genome.NumLink(); ++i) {
      if (genome.aLinkGenes[i].Enabled() == enabled) {
        linkIndexs.push_back(i);
      }
    }
    
    if (linkIndexs.size()>0) {
      ssize_t idx = linkIndexs[mlpack::math::RandInt(0, linkIndexs.size())];
      genome.aLinkGenes[idx].Enabled(!enabled);
    }
  }

  // Mutate: change single weight. Combine both biased and unbiased mutation.
  void MutateWeight(Genome& genome, double mutateProb, double perturbProb, double mutateSize) {
    double p = mlpack::math::Random();  // rand 0~1
    if (p > mutateProb) return;
    
    for (ssize_t i=0; i<genome.aLinkGenes.size(); ++i) {  
      double p2 = mlpack::math::Random();
      if (p2 < perturbProb) {  // Biased weight mutation.
        double deltaW = mlpack::math::RandNormal(0, mutateSize);
        double oldW = genome.aLinkGenes[i].Weight();
        genome.aLinkGenes[i].Weight(oldW + deltaW);
      } else {  // Unbiased weight mutation.
        double weight = mlpack::math::RandNormal(0, mutateSize);
        genome.aLinkGenes[i].Weight(weight);
      }
    }
  }

  // Compare which genome is better.
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

  // Crossover link weights. Assume momGenome is the better genome, childGenome is empty.
  // NOTICE: in the NEAT paper, disabled links also can crossover, calculate distance, etc.
  // Is it really a good idea???
  // If not, we will need to change CrossoverLinkAndNeuron, and Disjoint, and WeightDiff.
  void CrossoverLinkAndNeuron(Genome& momGenome, Genome& dadGenome, Genome& childGenome) {
    childGenome.NumInput(momGenome.NumInput());
    childGenome.NumOutput(momGenome.NumOutput());

    // Add input and output neuron genes to child genome.
    for (ssize_t i=0; i<(momGenome.NumInput() + momGenome.NumOutput()); ++i) {
      childGenome.aNeuronGenes.push_back(momGenome.aNeuronGenes[i]);
    }

    // Iterate to add link genes and neuron genes to child genome.
    for (ssize_t i=0; i<momGenome.NumLink(); ++i) {
      ssize_t innovId = momGenome.aLinkGenes[i].InnovationId();      
      ssize_t idx = dadGenome.GetLinkIndex(innovId);
      bool linkContainedInDad = (idx != -1);
      double randNum = mlpack::math::Random();

      // Exceed or disjoint link, add to child.
      if (!linkContainedInDad) {  
        childGenome.AddLink(momGenome.aLinkGenes[i]);

        // Add from neuron.
        ssize_t idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        ssize_t idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }
        continue;
      }

      // Common link in both parents, add mom's to child with probability 0.5.
      if (linkContainedInDad && randNum < 0.5) {
        childGenome.AddLink(momGenome.aLinkGenes[i]);

        // Add from neuron.
        ssize_t idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        ssize_t idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        if (idxInChild == -1) {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }
        continue;
      }

      // Common link in both parents, add dad's to child with probability 0.5.
      if (linkContainedInDad && randNum >= 0.5) {
        childGenome.AddLink(dadGenome.aLinkGenes[idx]);

        // Add from neuron.
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

  // Crossover two genome to get one genome.
  void Crossover(Genome& genome1, Genome& genome2, Genome& childGenome) {
    if (CompareGenome(genome1, genome2)) {  // genome1 is better
      CrossoverLinkAndNeuron(genome1, genome2, childGenome);
    } else {  // genome2 is better
      CrossoverLinkAndNeuron(genome2, genome1, childGenome);
    }
  }

  // Measure two genomes' disjoint (including exceed).
  // NOTICE: we can separate into disjoint and exceed. But currently maybe it is enough.
  double Disjoint(Genome& genome1, Genome& genome2) {
    double numDisjoint = 0;

    for (ssize_t i=0; i<genome1.NumLink(); ++i) {
      ssize_t innovId = genome1.aLinkGenes[i].InnovationId();
      bool linkContainedInGenome2 = genome2.ContainLink(innovId);
      if (!linkContainedInGenome2) {
        ++numDisjoint;
      } 
    }

    for (ssize_t i=0; i<genome2.NumLink(); ++i) {
      ssize_t innovId = genome2.aLinkGenes[i].InnovationId();
      bool linkContainedInGenome1 = genome1.ContainLink(innovId);
      if (!linkContainedInGenome1) {
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
      int linkEnabledInGenome1 = (int) genome1.aLinkGenes[i].Enabled();
      ssize_t innovId = genome1.aLinkGenes[i].InnovationId();
      ssize_t idx = genome2.GetLinkIndex(innovId);
      bool linkContainedInGenome2 = (idx != -1);

      if (linkContainedInGenome2) {
        int linkEnabledInGenome2 = (int) genome2.aLinkGenes[idx].Enabled();
        deltaW += std::abs(genome1.aLinkGenes[i].Weight() * linkEnabledInGenome1 - 
                           genome2.aLinkGenes[idx].Weight() * linkEnabledInGenome2);
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
  void AddGenomeToSpecies(Population& population, Genome& genome) {
    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      if (population.aSpecies[i].aGenomes.size() > 0) {
        if (IsSameSpecies(population.aSpecies[i].aGenomes[0], genome)) {  // each first genome in species is the representative genome.
          population.aSpecies[i].AddGenome(genome);
          return;
        }
      }
    }

    Species newSpecies = Species();
    newSpecies.AddGenome(genome);
    newSpecies.StaleAge(0);
    population.AddSpecies(newSpecies);
  }

  // Remove stale species.
  void RemoveStaleSpecies(Population& population) {
    for (std::vector<Species>::iterator it = population.aSpecies.begin();
         it != population.aSpecies.end();  /*it++*/) {
      if(it->StaleAge() > aStaleAgeThreshold) {
        it = population.aSpecies.erase(it);
      }
      else {
        ++it;
      }
    }
  }

  // Aggregate population's genomes.
  void AggregateGenomes(Population& population, std::vector<Genome>& genomes) {
    genomes.clear();
    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      for (ssize_t j=0; j<population.aSpecies[i].aGenomes.size(); ++j) {
        genomes.push_back(population.aSpecies[i].aGenomes[j]);
      }
    }
  }

  // Sort genomes by fitness. Smaller fitness is better and put first.
  void SortGenomes(std::vector<Genome>& genomes) {
    std::sort(genomes.begin(), genomes.end(), Species::CompareGenome);
  }

  // Get genome index in a genomes vector.
  ssize_t GetGenomeIndex(std::vector<Genome>& genomes, ssize_t id) {
    for (ssize_t i=0; i<genomes.size(); ++i) {
      if (genomes[i].Id() == id)
        return i;
    }
    return -1;
  }

  // Calculate species' average rank in population by fitness. Bigger is better.
  void CalcSpeciesAverageRank(Population& population, std::vector<double>& speciesAverageRank) {
    std::vector<Genome> genomes;
    AggregateGenomes(population, genomes);
    SortGenomes(genomes);
    speciesAverageRank.clear();

    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      double averageRank = 0;
      ssize_t speciesSize = population.aSpecies[i].aGenomes.size();
      for (ssize_t j=0; j<speciesSize; ++j) {
        averageRank += genomes.size() - GetGenomeIndex(genomes, population.aSpecies[i].aGenomes[j].Id());
      }
      averageRank = averageRank / speciesSize;
      speciesAverageRank.push_back(averageRank);
    }
  }

  // Remove weak species.
  void RemoveWeakSpecies(Population& population) {
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(population, speciesAverageRank);
    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);

    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      double weak = (std::floor(speciesAverageRank[i] * population.NumSpecies() / totalAverageRank)
                     < 1);  // NOTICE: here good or not?
      if (weak) {
        population.RemoveSpecies(i);
      }
    }
  }

  // Remove empty species.
  void RemoveEmptySpecies(Population& population) {
    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      if (population.aSpecies[i].aGenomes.size() == 0) {
        population.aSpecies.erase(population.aSpecies.begin() + i);
      }
    }
  }

  // Remove a portion weak genomes in each species
  void CullSpecies(Population& population, double percentageToRemove) {
    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      population.aSpecies[i].SortGenomes();
      ssize_t numRemove = std::floor(population.aSpecies[i].aGenomes.size() * percentageToRemove);
      while (numRemove > 0) {
        population.aSpecies[i].aGenomes.pop_back();
        --numRemove;
      }
    }
    RemoveEmptySpecies(population);
  }

  // Only keep the best genome in each species.
  void CullSpeciesToOne(Population& population) {
    for (ssize_t i=0; i<population.aSpecies.size(); ++i) {
      population.aSpecies[i].SortGenomes();
      ssize_t speciesSize = population.aSpecies[i].aGenomes.size();
      if (speciesSize > 0) {
        Genome bestGenome = population.aSpecies[i].aGenomes[0];
        population.aSpecies[i].aGenomes.clear();
        population.aSpecies[i].aGenomes.push_back(bestGenome);
      }
    }
    RemoveEmptySpecies(population);
  }

  // Mutate child by combining different mutations.
  // NOTICE: how we organize different mutations is kind of flexible.
  void Mutate(Genome& genome) {
    // Mutate weights.
    MutateWeight(genome, aMutateWeightProb, aPerturbWeightProb, aMutateWeightSize);

    // Mutate add link.
    double p = aMutateAddLinkProb;
    while (p > 0) {  // so p can be bigger than 1 and mutate can happen multiple times.
      if (mlpack::math::Random() < p) {
        MutateAddLink(genome, aMutateAddLinkProb, aMutateAddRecurrentLinkProb, aMutateAddLoopLinkProb);
      }
      --p;
    }

    // Mutate add neuron.
    p = aMutateAddNeuronProb;
    while (p > 0) {
      if (mlpack::math::Random() < p) {
        MutateAddNeuron(genome, aMutateAddNeuronProb);
      }
      --p;
    }

    // Mutate enabled node to disabled.
    p = aMutateEnabledProb;
    while (p > 0) {
      if (mlpack::math::Random() < p) {
        MutateEnableDisable(genome, true, aMutateEnabledProb);
      }
      --p;
    }

    // Mutate disabled node to enabled.
    p = aMutateDisabledProb;
    while (p > 0) {
      if (mlpack::math::Random() < p) {
        MutateEnableDisable(genome, false, aMutateDisabledProb);
      }
      --p;
    }
  }

  // Breed child for a species.
  // Crossover to born a child, or copy a child, then mutate it.
  // NOTICE: can have different ways to breed a child.
  bool BreedChild(Species& species, Genome& childGenome, double crossoverProb) {
    double p = mlpack::math::Random();
    ssize_t speciesSize = species.aGenomes.size();

    if (speciesSize == 0)
      return false;

    if (p < crossoverProb) {
      ssize_t idx1 = mlpack::math::RandInt(0, speciesSize);
      ssize_t idx2 = mlpack::math::RandInt(0, speciesSize);
      if (idx1 != idx2) {
        Crossover(species.aGenomes[idx1], species.aGenomes[idx2], childGenome);
      } else {
        return false;
      }
    } else {
      ssize_t idx = mlpack::math::RandInt(0, speciesSize);
      childGenome = species.aGenomes[idx];
    }

    Mutate(childGenome);

    return true;
  }


  // Initialize population.
  void InitPopulation() {
    aPopulation = Population(aSeedGenome, aPopulationSize);
  }

  // Reproduce next generation of population.
  void Reproduce() {
    // keep best genome.
    Genome lastBestGenome = aPopulation.BestGenome();

    // Remove weak genomes in each species.
    CullSpecies(aPopulation, aCullSpeciesPercentage);

    // Remove stale species, weak species.
    if (aPopulation.aSpecies.size() > 10) {
      RemoveStaleSpecies(aPopulation);
      //RemoveWeakSpecies(aPopulation);
    }

    // Breed children in each species. 
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(aPopulation, speciesAverageRank);

    //DEBUGGING!!!!!!!!!
    printf("Species average rank are: ");
    for (ssize_t s=0; s<speciesAverageRank.size(); ++s) {
      std::cout<< speciesAverageRank[s] << "  ";
    }
    printf("\n");
    //DEBUGGING!!!!!!!!!

    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);
    std::vector<Genome> childGenomes;
    for (ssize_t i=0; i<aPopulation.aSpecies.size(); ++i) {
      // number of child genomes by this species.
      ssize_t numBreed = std::floor(speciesAverageRank[i] * aPopulationSize / totalAverageRank) - 1;
      ssize_t numBreedSuccess = 0;
      while (numBreedSuccess < numBreed) {
        Genome genome;
        bool hasBaby = BreedChild(aPopulation.aSpecies[i], genome, aCrossoverRate);
        if (hasBaby) {
          childGenomes.push_back(genome);
          ++numBreedSuccess;
        }
      }
    }

    // Keep the best in each species.
    CullSpeciesToOne(aPopulation);  // NOTICE: I found that this makes XOR test converges fast.

    // Random choose species and breed child until reach population size.
    childGenomes.push_back(lastBestGenome);
    ssize_t currentNumGenome = childGenomes.size() + aPopulation.PopulationSize();
    while (currentNumGenome < aPopulationSize) {
      ssize_t speciesIndex = mlpack::math::RandInt(0, aPopulation.aSpecies.size());
      Genome genome;
      bool hasBaby = BreedChild(aPopulation.aSpecies[speciesIndex], genome, aCrossoverRate);
      if (hasBaby) {
        childGenomes.push_back(genome);
        ++currentNumGenome;
      }
    }

    // Speciate genomes into new species.
    std::vector<Genome> populationGenomes;
    AggregateGenomes(aPopulation, populationGenomes);
    aPopulation.aSpecies.clear();
    for (ssize_t i=0; i<populationGenomes.size(); ++i) {
      AddGenomeToSpecies(aPopulation, populationGenomes[i]);
    }
    for (ssize_t i=0; i<childGenomes.size(); ++i) {
      AddGenomeToSpecies(aPopulation, childGenomes[i]);
    }

    //DEBUGGING!!!!!!!!!
    printf("Species sizes are: ");
    for (ssize_t s=0; s<aPopulation.aSpecies.size(); ++s) {
      std::cout<< aPopulation.aSpecies[s].aGenomes.size() << "  ";
    }
    printf("\n");
    //DEBUGGING!!!!!!!!!

    // Reassign genome IDs.
    aPopulation.ReassignGenomeId();
  }

  // Evaluate genomes in population.
  // Set genomes' fitness, species' and population's best fitness and genome.
  void Evaluate() {
    for (ssize_t i=0; i<aPopulation.aSpecies.size(); ++i) {
      for (ssize_t j=0; j<aPopulation.aSpecies[i].aGenomes.size(); ++j) {
        double fitness = aTask.EvalFitness(aPopulation.aSpecies[i].aGenomes[j]);
        aPopulation.aSpecies[i].aGenomes[j].Fitness(fitness);
      }

      double oldSpeciesBestFitness = aPopulation.aSpecies[i].BestFitness();
      aPopulation.aSpecies[i].SetBestFitnessAndGenome();
      double newSpeciesBestFitness = aPopulation.aSpecies[i].BestFitness();
      if (newSpeciesBestFitness < oldSpeciesBestFitness) {
        aPopulation.aSpecies[i].StaleAge(0);
      } else {
        ssize_t staleAge = aPopulation.aSpecies[i].StaleAge();
        aPopulation.aSpecies[i].StaleAge(staleAge + 1);
      }
    }
    aPopulation.SetBestFitnessAndGenome();
  }

  // Evolve.
  void Evolve() {
    // Generate initial species at random.
    ssize_t generation = 0;
    InitPopulation();

    // Speciate genomes into species.
    std::vector<Genome> genomes;
    AggregateGenomes(aPopulation, genomes);
    aPopulation.aSpecies.clear();
    for (ssize_t i=0; i<genomes.size(); ++i) {
      AddGenomeToSpecies(aPopulation, genomes[i]);
    }
    
    // Repeat
    while (generation < aMaxGeneration) {
      // Evaluate all genomes in population.
      Evaluate();

      // Output some information.
      printf("Generation: %zu\tBest fitness: %f\n", generation, aPopulation.BestFitness());

      // Reproduce next generation.
      Reproduce();
      ++generation;
    }
  }
  
 private:
  // Task.
  TaskType aTask;

  // Seed genome. It is used for init population.
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

  // Threshold for species stale age.
  ssize_t aStaleAgeThreshold;

  // Crossover rate.
  double aCrossoverRate;

  // Percentage to remove in each species.
  double aCullSpeciesPercentage;

  // Probability to mutate a genome's weight
  double aMutateWeightProb;

  // Probability to mutate a genome's weight in biased way (add Gaussian perturb noise).
  double aPerturbWeightProb;

  // The Gaussian noise variance when mutating genome weights.
  double aMutateWeightSize;

  // Probability to add link to genome.
  double aMutateAddLinkProb;

  // Probability to add a recurrent link.
  double aMutateAddRecurrentLinkProb;

  // Probability to add a loop link.
  double aMutateAddLoopLinkProb;

  // Probability to add neuron to genome.
  double aMutateAddNeuronProb;

  // Probability to turn enabled link to disabled.
  double aMutateEnabledProb;

  // Probability to turn disabled link to enabled.
  double aMutateDisabledProb;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEAT_HPP
