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
  // TODO: make innovation a class and pass as parameter? Also for other similar functions.
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

  // Mutate: enable disabled, or disable enabled link.
  void MutateEnableDisable(Genome& genome, bool enabled, double mutateProb) {
    std::vector<ssize_t> linkIndexs;
    for (ssize_t i=0; i<genome.NumLink(); ++i) {
      if (genome.aLinkGenes[i].Enabled() == enabled) {
        linkIndexs.push_back(i);
      }
    }

    double p = mlpack::math::Random();
    if (p<mutateProb && linkIndexs.size()>0) {
      ssize_t idx = linkIndexs[mlpack::math::RandInt(0, linkIndexs.size())];
      genome.aLinkGenes[idx].Enabled(!enabled);  // Reverse enabled status to opposite.
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

  // Crossover link weights.
  // NOTICE: assume momGenome is the better genome.
  // NOTICE: assume childGenome is empty.
  // NOTICE: in the NEAT paper, disabled links also can crossover, calculate distance, etc.
  // Is it really a good idea???
  // If not, we will need to change CrossoverLinkAndNeuron, and Disjoint, and WeightDiff.
  void CrossoverLinkAndNeuron(Genome& momGenome, Genome& dadGenome, Genome& childGenome) {
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

      if (!linkContainedInDad) {  // exceed or disjoint
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

      if (linkContainedInDad && randNum < 0.5) {
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

      if (linkContainedInDad && randNum >= 0.5) {
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

  // Crossover two genome to get one genome.
  void Crossover(Genome& genome1, Genome& genome2, Genome& childGenome) {
    if (CompareGenome(genome1, genome2)) {  // genome1 is better
      CrossoverLinkAndNeuron(genome1, genome2, childGenome);
    } else {
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
  // TODO: what if one or two weights are disabled? 0 or still the weight?
  double WeightDiff(Genome& genome1, Genome& genome2) {
    double deltaW = 0;
    ssize_t coincident = 0;

    for (ssize_t i=0; i<genome1.NumLink(); ++i) {
      ssize_t innovId = genome1.aLinkGenes[i].InnovationId();
      ssize_t idx = genome2.GetLinkIndex(innovId);
      bool linkContainedInGenome2 = (idx != -1);
      if (linkContainedInGenome2) {
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
  void AddGenomeToSpecies(Population& population, Genome& genome) {
    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      if (population.aSpecies[i].SpeciesSize() > 0) {
        if (IsSameSpecies(population.aSpecies[i].aGenomes[0], genome)) {  // each first genome in species is the representative genome.
          population.aSpecies[i].AddGenome(genome);
          return;
        }
      }
    }

    Species newSpecies = Species();
    newSpecies.AddGenome(genome);
    newSpecies.Id(population.NextSpeciesId());  // NOTICE: changed species id.
    newSpecies.StaleAge(0);
    population.AddSpecies(newSpecies);
  }

  // Remove stale species.
  void RemoveStaleSpecies(Population& population) {
    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      if (population.aSpecies[i].StaleAge() > aStaleAgeThreshold) {
        population.RemoveSpecies(i);
      }
    }
  }

  // Set adjusted fitness.
  // NOTICE: we assume fitness have already evaluated before adjust it.
  // Maybe we can add some flag or other way to judge whether is evaluated or not.
  void AdjustFitness(Population& population) {
    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      if (population.aSpecies[i].SpeciesSize() > 0) {
        for (ssize_t j=0; j<population.aSpecies[i].SpeciesSize(); ++j) {
          double fitness = population.aSpecies[i].aGenomes[j].Fitness();
          ssize_t speciesSize = population.aSpecies[i].SpeciesSize();
          double adjustedFitness = fitness / speciesSize;
          population.aSpecies[i].aGenomes[j].AdjustedFitness(adjustedFitness);
        }
      }
    }
  }

  // Aggregate population's genomes.
  void AggregateGenomes(Population& population, std::vector<Genome>& genomes) {
    genomes.clear();
    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      for (ssize_t j=0; j<population.aSpecies[i].SpeciesSize(); ++j) {
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

  // Calculate species' average rank in population.
  void CalcSpeciesAverageRank(Population& population, std::vector<double>& speciesAverageRank) {
    std::vector<Genome> genomes;
    AggregateGenomes(population, genomes);
    SortGenomes(genomes);
    speciesAverageRank.clear();

    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      double averageRank = 0;
      ssize_t speciesSize = population.aSpecies[i].SpeciesSize();

      for (ssize_t j=0; j<speciesSize; ++j) {
        averageRank += GetGenomeIndex(genomes, population.aSpecies[i].aGenomes[j].Id());
      }

      averageRank = averageRank / speciesSize;  // smaller is better.
      speciesAverageRank.push_back(averageRank);
    }
  }

  // Remove weak species.
  void RemoveWeakSpecies(Population& population) {
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(population, speciesAverageRank);
    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);

    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      double weak = (std::floor(speciesAverageRank[i] * population.PopulationSize() / totalAverageRank)
                    > 1);
      if (weak) {
        population.RemoveSpecies(i);
      }
    }
  }

  // Remove a portion weak genomes in each species
  void CullSpecies(Population& population, double percentageToRemove) {
    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      population.aSpecies[i].SortGenomes();
      ssize_t numRemove = std::floor(population.aSpecies[i].SpeciesSize() * percentageToRemove);
      while (numRemove > 0) {
        population.aSpecies[i].aGenomes.pop_back();
        --numRemove;
      }
    }
  }

  // Only keep the best genome in each species.
  void CullSpeciesToOne(Population& population) {
    for (ssize_t i=0; i<population.NumSpecies(); ++i) {
      population.aSpecies[i].SortGenomes();
      ssize_t speciesSize = population.aSpecies[i].SpeciesSize();
      if (speciesSize > 0) {
        Genome bestGenome = population.aSpecies[i].aGenomes[0];
        population.aSpecies[i].aGenomes.clear();
        population.aSpecies[i].aGenomes.push_back(bestGenome);
      }
    }
  }

  // Mutate child by different mutations.
  // NOTICE: how we organize different mutations is kind of flexible.
  void Mutate(Genome& genome) {
    // NOTICE: we can change mutate rates here. Randomly let mutate rates be bigger or smaller.

    // Mutate weights.
    MutateWeight(genome, aMutateWeightProb, aPerturbWeightProb, aMutateWeightSize);

    // Mutate link. TODO: check link mutate implementation is correct or not.
    double p = aMutateAddLinkProb;
    while (p > 0) {  // so p can be bigger than 1 and mutate can happen multiple times.
      if (mlpack::math::Random() < p) {
        MutateAddLink(genome, aMutateAddLinkProb);
      }
      --p;
    }

    // Mutate neuron
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
  // NOTICE: can have different ways to breed a child.
  void BreedChild(Species& species, Genome& childGenome, double crossoverProb) {
    double p = mlpack::math::Random();
    ssize_t speciesSize = species.aGenomes.size();
    if (p < crossoverProb) {
      ssize_t idx1 = mlpack::math::RandInt(0, speciesSize);
      ssize_t idx2 = mlpack::math::RandInt(0, speciesSize);
      Crossover(species.aGenomes[idx1], species.aGenomes[idx2], childGenome);
    } else {
      ssize_t idx = mlpack::math::RandInt(0, speciesSize);
      childGenome = species.aGenomes[idx];
    }

    Mutate(childGenome);
  }


  // Initialize population.
  void InitPopulation() {
    aPopulation = Population(aSeedGenome, aPopulationSize);
  }

  // Reproduce next generation of population.
  void Reproduce() {
    // Remove stale species.
    RemoveStaleSpecies(aPopulation);

    // Remove weak genomes in each species.
    CullSpecies(aPopulation, aCullSpeciesPercentage);

    // Remove weak species.
    RemoveWeakSpecies(aPopulation);

    // Breed children in each species. 
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(aPopulation, speciesAverageRank);
    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);
    std::vector<Genome> childGenomes;
    for (ssize_t i=0; i<aPopulation.aSpecies.size(); ++i) {
      // number of child genomes by this species.
      ssize_t numBreed = std::floor(speciesAverageRank[i] * aPopulationSize / totalAverageRank) - 1;

      for (ssize_t j=0; j<numBreed; ++j) {
        Genome genome; //!!!!!!!!!!!!!
        BreedChild(aPopulation.aSpecies[i], genome, aCrossoverRate);
        childGenomes.push_back(genome);
      }
    }

    // Keep the best in each species.
    CullSpeciesToOne(aPopulation);

    // Random choose species and breed child until reach population size.
    while (childGenomes.size() + aPopulation.aSpecies.size() < aPopulationSize) {
      ssize_t speciesIndex = mlpack::math::RandInt(0, aPopulation.aSpecies.size());
      Genome genome;  //!!!!!!!!!!!!!
      BreedChild(aPopulation.aSpecies[speciesIndex], genome, aCrossoverRate);
      childGenomes.push_back(genome);
    }

    // Speciate genomes into new species.
    for (ssize_t i=0; i<childGenomes.size(); ++i) {
      AddGenomeToSpecies(aPopulation, childGenomes[i]);
    }

    // Reassign genome IDs.
    aPopulation.ReassignGenomeId();
  }

  // Evaluate genomes in population, set genomes' fitness.
  void Evaluate() {
    for (ssize_t i=0; i<aPopulation.NumSpecies(); ++i) {
      for (ssize_t j=0; j<aPopulation.aSpecies[i].SpeciesSize(); ++j) {
        double fitness = aTask.EvalFitness(aPopulation.aSpecies[i].aGenomes[j]);
        aPopulation.aSpecies[i].aGenomes[j].Fitness(fitness);
      }

      double oldSpeciesBestFitness = aPopulation.aSpecies[i].BestFitness();
      aPopulation.aSpecies[i].SetBestFitness();
      double newSpeciesBestFitness = aPopulation.aSpecies[i].BestFitness();
      if (newSpeciesBestFitness < oldSpeciesBestFitness) {
        aPopulation.aSpecies[i].StaleAge(0);
      } else {
        ssize_t staleAge = aPopulation.aSpecies[i].StaleAge();
        aPopulation.aSpecies[i].StaleAge(staleAge + 1);
      }
    }
    aPopulation.SetBestFitness();
  }

  // Evolve.
  void Evolve() {
    // Generate initial species at random.
    ssize_t generation = 0;
    InitPopulation();
    
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
