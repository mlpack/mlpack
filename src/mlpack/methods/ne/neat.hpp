 /**
 * @file neat.hpp
 * @author Bang Liu
 *
 * Definition of NEAT class.
 */
#ifndef MLPACK_METHODS_NE_NEAT_HPP
#define MLPACK_METHODS_NE_NEAT_HPP

#include <cstdio>
#include <numeric>
#include <algorithm>

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

/**
 * Structure to save link innovation.
 *
 * This structure saves the new type of links created during evolution.
 * So that same innovation will get same link innovation id, and it helps 
 * to align links for crossover even when network structures are different.
 */
struct LinkInnovation
{
  int fromNeuronId;
  int toNeuronId;
  int newLinkInnovId;
};

/**
 * Structure to save neuron innovation.
 *
 * This structure saves the new type of neurons created during evolution.
 * So that same innovation will get same neuron id, and it helps 
 * to align links for crossover even when network structures are different.
 */
struct NeuronInnovation
{
  int splitLinkInnovId;
  ActivationFuncType actFuncType;
  int newNeuronId;
  int newInputLinkInnovId;
  int newOutputLinkInnovId;
};

/**
 * This is enumeration of link types.
 */
enum LinkType
{
  FORWARD_LINK = 0,
  BACKWARD_LINK,
  RECURRENT_LINK,
  BIAS_LINK
};

/**
 * This class implements  NEAT algorithm.
 */
template<typename TaskType>
class NEAT
{
 public:
  //! Task to solve.
  TaskType aTask;

  //! Seed genome. It is used for init population.
  Genome aSeedGenome;

  //! Population to evolve.
  Population aPopulation;

  //! Population size.
  int aPopulationSize;

  //! List of link innovations.
  std::vector<LinkInnovation> aLinkInnovations;

  //! List of neuron innovations.
  std::vector<NeuronInnovation> aNeuronInnovations;

  //! Next neuron id.
  int aNextNeuronId;

  //! Next link id.
  int aNextLinkInnovId;

  //! Max number of generation to evolve.
  int aMaxGeneration;

  //! Efficient for disjoint.
  double aCoeffDisjoint;

  //! Efficient for weight difference.
  double aCoeffWeightDiff;

  //! Threshold for judge whether belong to same species.
  double aCompatThreshold;

  //! Threshold for species stale age.
  int aStaleAgeThreshold;

  //! Crossover rate.
  double aCrossoverRate;

  //! Percentage to remove in each species.
  double aCullSpeciesPercentage;

  //! Probability to mutate a genome's weight.
  double aMutateWeightProb;

  //! Probability to mutate a genome's weight in biased way (add Gaussian perturb noise).
  double aPerturbWeightProb;

  //! The Gaussian noise variance when mutating genome weights.
  double aMutateWeightSize;

  //! Probability to add a forward link.
  double aMutateAddForwardLinkProb;

  //! Probability to add a backward link.
  double aMutateAddBackwardLinkProb;

  //! Probability to add a recurrent link.
  double aMutateAddRecurrentLinkProb;

  //! Probability to add a bias link.
  double aMutateAddBiasLinkProb;

  //! Probability to add neuron to genome.
  double aMutateAddNeuronProb;

  //! Probability to turn enabled link to disabled.
  double aMutateEnabledProb;

  //! Probability to turn disabled link to enabled.
  double aMutateDisabledProb;

  //! Species number threshold.
  int aNumSpeciesThreshold;

  /**
   * Default constructor.
   */
  NEAT()
  {
    mlpack::math::RandomSeed(1);

    // Set NEAT algorithm parameters.
    Parameters params;
    params.aPopulationSize = 500;
    params.aMaxGeneration = 500;
    params.aCoeffDisjoint = 2.0;
    params.aCoeffWeightDiff = 0.4;
    params.aCompatThreshold = 1.0;
    params.aStaleAgeThreshold = 15;
    params.aCrossoverRate = 0.75;
    params.aCullSpeciesPercentage = 0.5;
    params.aMutateWeightProb = 0.2;
    params.aPerturbWeightProb = 0.9;
    params.aMutateWeightSize = 0.1;
    params.aMutateAddForwardLinkProb = 0.9;
    params.aMutateAddBackwardLinkProb = 0;
    params.aMutateAddRecurrentLinkProb = 0;
    params.aMutateAddBiasLinkProb = 0;
    params.aMutateAddNeuronProb = 0.6;
    params.aMutateEnabledProb = 0.2;
    params.aMutateDisabledProb = 0.2;
    params.aNumSpeciesThreshold = 10;

    // Construct seed genome for xor task.
    int id = 0;
    int numInput = 3;
    int numOutput = 1;
    double fitness = -1;
    std::vector<NeuronGene> neuronGenes;
    std::vector<LinkGene> linkGenes;

    NeuronGene inputGene1(0, INPUT, LINEAR, 0, std::vector<double>(), 0, 0);
    NeuronGene inputGene2(1, INPUT, LINEAR, 0, std::vector<double>(), 0, 0);
    NeuronGene biasGene(2, BIAS, LINEAR, 0, std::vector<double>(), 0, 0);
    NeuronGene outputGene(3, OUTPUT, SIGMOID, 1, std::vector<double>(), 0, 0);
    NeuronGene hiddenGene(4, HIDDEN, SIGMOID, 0.5, std::vector<double>(), 0, 0);

    neuronGenes.push_back(inputGene1);
    neuronGenes.push_back(inputGene2);
    neuronGenes.push_back(biasGene);
    neuronGenes.push_back(outputGene);
    neuronGenes.push_back(hiddenGene);

    LinkGene link1(0, 3, 0, 0, true);
    LinkGene link2(1, 3, 1, 0, true);
    LinkGene link3(2, 3, 2, 0, true);
    LinkGene link4(0, 4, 3, 0, true);
    LinkGene link5(1, 4, 4, 0, true);
    LinkGene link6(2, 4, 5, 0, true);
    LinkGene link7(4, 3, 6, 0, true);

    linkGenes.push_back(link1);
    linkGenes.push_back(link2);
    linkGenes.push_back(link3);
    linkGenes.push_back(link4);
    linkGenes.push_back(link5);
    linkGenes.push_back(link6);
    linkGenes.push_back(link7);

    Genome seedGenome = Genome(0, 
                               neuronGenes,
                               linkGenes,
                               numInput,
                               numOutput,
                               fitness);

    // Set neat members. (Except task.)
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
    aMutateAddForwardLinkProb = params.aMutateAddForwardLinkProb;
    aMutateAddBackwardLinkProb = params.aMutateAddBackwardLinkProb;
    aMutateAddRecurrentLinkProb = params.aMutateAddRecurrentLinkProb;
    aMutateAddBiasLinkProb = params.aMutateAddBiasLinkProb;
    aMutateAddNeuronProb = params.aMutateAddNeuronProb;
    aMutateEnabledProb = params.aMutateEnabledProb;
    aMutateDisabledProb = params.aMutateDisabledProb;
    aNumSpeciesThreshold = params.aNumSpeciesThreshold;
  }

  /**
   * Parametric constructor.
   *
   * @param task The task to solve.
   * @param seedGenome The genome to initialize population.
   * @param params The Parameter object that contains algorithm parameters.
   */
  NEAT(TaskType task, Genome& seedGenome, Parameters& params)
  {
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
    aMutateAddForwardLinkProb = params.aMutateAddForwardLinkProb;
    aMutateAddBackwardLinkProb = params.aMutateAddBackwardLinkProb;
    aMutateAddRecurrentLinkProb = params.aMutateAddRecurrentLinkProb;
    aMutateAddBiasLinkProb = params.aMutateAddBiasLinkProb;
    aMutateAddNeuronProb = params.aMutateAddNeuronProb;
    aMutateEnabledProb = params.aMutateEnabledProb;
    aMutateDisabledProb = params.aMutateDisabledProb;
    aNumSpeciesThreshold = params.aNumSpeciesThreshold;
  }

  /**
   * Destructor.
   */
  ~NEAT() {}

  /**
   * Check whether a link innovation already exist.
   *
   * @param fromNeuronId The from neuron's id of the link.
   * @param toNeuronId The to neuron's id of the link.
   */
  int CheckLinkInnovation(int fromNeuronId, int toNeuronId)
  {
    for (int i=0; i<aLinkInnovations.size(); ++i)
    {
      if (aLinkInnovations[i].fromNeuronId == fromNeuronId && 
          aLinkInnovations[i].toNeuronId == toNeuronId)
      {
        return i;
      }
    }
    
    return -1;  // -1 means no match found, a new innovation.
  }

  /**
   * Check whether a neuron innovation already exist.
   *
   * @param splitLinkInnovId The innovation id of the link to be split.
   */
  int CheckNeuronInnovation(int splitLinkInnovId, ActivationFuncType actFuncType)
  {
    for (int i=0; i<aNeuronInnovations.size(); ++i)
    {
      if (aNeuronInnovations[i].splitLinkInnovId == splitLinkInnovId &&
          aNeuronInnovations[i].actFuncType == actFuncType)
      {
        return i;
      }
    }

    return -1;
  }

  /**
   * Add a new link innovation to link innovation list.
   *
   * @param fromNeuronId The from neuron id of new link.
   * @param toNeuronId The to neuron id of new link.
   * @param linkInnov The new link innovation to add.
   */
  void AddLinkInnovation(int fromNeuronId, int toNeuronId, LinkInnovation& linkInnov)
  {
    linkInnov.fromNeuronId = fromNeuronId;
    linkInnov.toNeuronId = toNeuronId;
    linkInnov.newLinkInnovId = aNextLinkInnovId++;
    aLinkInnovations.push_back(linkInnov);
  }

  /**
   * Add a new neuron innovation to neuron innovation list.
   *
   * @param splitLinkInnovId The id of link that been split to add new neuron.
   * @param neuronInnov The neuron innovation to add.
   */
  void AddNeuronInnovation(int splitLinkInnovId, NeuronInnovation& neuronInnov)
  {
    neuronInnov.splitLinkInnovId = splitLinkInnovId;
    neuronInnov.newNeuronId = aNextNeuronId++;
    neuronInnov.newInputLinkInnovId = aNextLinkInnovId++;
    neuronInnov.newOutputLinkInnovId = aNextLinkInnovId++;
    aNeuronInnovations.push_back(neuronInnov);
  }

  /**
   * Check if link exist in genome or not.
   *
   * @param genome The genome to check.
   * @param fromNeuronId The from neuron id of the link.
   * @param toNeuronId The to neuron id of the link.
   */
  int IsLinkExist(Genome& genome, int fromNeuronId, int toNeuronId)
  {
    for (int i=0; i<genome.NumLink(); ++i)
    {
      if (genome.aLinkGenes[i].FromNeuronId() == fromNeuronId &&
          genome.aLinkGenes[i].ToNeuronId() == toNeuronId)
      {
        return i;
      }
    }
    return -1;  // -1 means not exist.
  } 

  /**
   * Mutate: add new link to genome.
   *
   * @param genome Add new link to this genome.
   * @param linkType The type of the new link: 
   *        FORWARD_LINK, BACKWARD_LINK, BIAS_LINK, RECURRENT_LINK.
   * @param mutateAddLinkProb The probability of adding a specific type of link.
   */
  void MutateAddLink(Genome& genome,
                     LinkType linkType,
                     double mutateAddLinkProb)
  {
    // Whether mutate or not.
    double p = mlpack::math::Random();
    if (p > mutateAddLinkProb) return;

    if (genome.aNeuronGenes.size() == 0) return;

    // Select from neuron and to neuron.
    int fromNeuronIdx = -1;
    int fromNeuronId = -1;
    int toNeuronIdx = -1;
    int toNeuronId = -1;

    switch (linkType)
    {
      case FORWARD_LINK:
        // Select from neuron.
        fromNeuronIdx = mlpack::math::RandInt(0, genome.aNeuronGenes.size());
        fromNeuronId = genome.aNeuronGenes[fromNeuronIdx].Id();

        // Select to neuron which cannot be input.
        toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.aNeuronGenes.size()); 
        toNeuronId = genome.aNeuronGenes[toNeuronIdx].Id();

        // Don't allow same depth connection.
        if (genome.aNeuronGenes[fromNeuronIdx].Depth() == genome.aNeuronGenes[toNeuronIdx].Depth())
        {
          return;
        }

        // Swap if backward.
        if (genome.aNeuronGenes[fromNeuronIdx].Depth() > genome.aNeuronGenes[toNeuronIdx].Depth())
        {
          std::swap(fromNeuronIdx, toNeuronIdx);
          std::swap(fromNeuronId, toNeuronId);
        }

        break;
      case BACKWARD_LINK:
        // Select from neuron.
        fromNeuronIdx = mlpack::math::RandInt(0, genome.aNeuronGenes.size());
        fromNeuronId = genome.aNeuronGenes[fromNeuronIdx].Id();

        // Select to neuron which cannot be input.
        toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.aNeuronGenes.size()); 
        toNeuronId = genome.aNeuronGenes[toNeuronIdx].Id();

        // Don't allow same depth connection.
        if (genome.aNeuronGenes[fromNeuronIdx].Depth() == genome.aNeuronGenes[toNeuronIdx].Depth())
        {
          return;
        }

        // Swap if forward.
        if (genome.aNeuronGenes[fromNeuronIdx].Depth() < genome.aNeuronGenes[toNeuronIdx].Depth()) 
        {
          std::swap(fromNeuronIdx, toNeuronIdx);
          std::swap(fromNeuronId, toNeuronId);
        }

        break;
      case RECURRENT_LINK:
        // Select recurrent neuron.
        fromNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.aNeuronGenes.size());
        fromNeuronId = genome.aNeuronGenes[fromNeuronIdx].Id();
        toNeuronIdx = fromNeuronIdx;
        toNeuronId = fromNeuronId;
        break;
      case BIAS_LINK:
        // Set from neuron as the BIAS neuron.
        fromNeuronIdx = genome.NumInput() - 1;
        fromNeuronId = genome.aNeuronGenes[fromNeuronIdx].Id();

        // Select to neuron which cannot be input.
        toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.aNeuronGenes.size()); 
        toNeuronId = genome.aNeuronGenes[toNeuronIdx].Id();
        break;
      default:
        return;
    }

    // Check link already exist or not.
    int linkIdx = IsLinkExist(genome, fromNeuronId, toNeuronId);
    if (linkIdx != -1)
    {
      genome.aLinkGenes[linkIdx].Enabled(true);
      return;
    }

    // Check innovation already exist or not.
    int innovIdx = CheckLinkInnovation(fromNeuronId, toNeuronId);
    if (innovIdx != -1)
    {
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

  /**
   * Mutate: add new neuron to genome.
   *
   * @param genome Add neuron to this genome.
   * @param mutateAddNeuronProb The probability to add new neuron to a genome.
   * @param randomActFuncType Whether the activation function of new neuron is random or not.
   */
  void MutateAddNeuron(Genome& genome, double mutateAddNeuronProb, bool randomActFuncType)
  {
    // Whether mutate or not.
    double p = mlpack::math::Random();
    if (p > mutateAddNeuronProb) return;

    // No link.
    if (genome.NumLink() == 0) return;

    // Select link to split.
    int linkIdx = mlpack::math::RandInt(0, genome.NumLink());
    if (!genome.aLinkGenes[linkIdx].Enabled()) return;

    genome.aLinkGenes[linkIdx].Enabled(false);
    NeuronGene fromNeuron;
    genome.GetNeuronById(genome.aLinkGenes[linkIdx].FromNeuronId(), fromNeuron);
    NeuronGene toNeuron;
    genome.GetNeuronById(genome.aLinkGenes[linkIdx].ToNeuronId(), toNeuron);

    // Check innovation already exist or not.
    int splitLinkInnovId = genome.aLinkGenes[linkIdx].InnovationId();
    ActivationFuncType actFuncType = SIGMOID;
    if (randomActFuncType) 
    {
      actFuncType = static_cast<ActivationFuncType>(rand() % ActivationFuncType::COUNT);
    }
    int innovIdx = CheckNeuronInnovation(splitLinkInnovId, actFuncType);

    // If existing innovation.
    if (innovIdx != -1)
    {
      // Check whether this genome already contains the neuron.
      int neuronIdx = genome.GetNeuronIndex(aNeuronInnovations[innovIdx].newNeuronId);
      if (neuronIdx != -1)  // The neuron already exist.
      {
        // Enable the input and output link.
        int inputLinkIdx = genome.GetLinkIndex(aNeuronInnovations[innovIdx].newInputLinkInnovId);
        int outputLinkIdx = genome.GetLinkIndex(aNeuronInnovations[innovIdx].newOutputLinkInnovId);
        genome.aLinkGenes[inputLinkIdx].Enabled(true);
        genome.aLinkGenes[outputLinkIdx].Enabled(true);
      } else
      {
        NeuronGene neuronGene(aNeuronInnovations[innovIdx].newNeuronId,
                              HIDDEN,
                              actFuncType,
                              (fromNeuron.Depth() + toNeuron.Depth()) / 2,
                              std::vector<double>(),
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
      }
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
                          actFuncType,
                          (fromNeuron.Depth() + toNeuron.Depth()) / 2,
                          std::vector<double>(),
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

  /**
   * Mutate: enable disabled link, or disable enabled link.
   *
   * @param genome The genome to apply this mutation operator.
   * @param enabled If true, turn enabled link to disabled;
   *        If false, turn disabled link to enabled.
   * @param mutateProb The probability to reverse link's enabled status.
   */
  void MutateEnableDisable(Genome& genome, bool enabled, double mutateProb)
  {
    double p = mlpack::math::Random();
    if (p > mutateProb) return;

    std::vector<int> linkIndexs;
    for (int i=0; i<genome.NumLink(); ++i)
    {
      if (genome.aLinkGenes[i].Enabled() == enabled)
      {
        linkIndexs.push_back(i);
      }
    }
    
    if (linkIndexs.size()>0)
    {
      int idx = linkIndexs[mlpack::math::RandInt(0, linkIndexs.size())];
      genome.aLinkGenes[idx].Enabled(!enabled);
    }
  }

  /**
   * Mutate: change single weight. Combine both biased and unbiased mutation.
   *
   * @param genome The genome to apply this mutation operator.
   * @param mutateProb The probability to mutate a genome's weights.
   * @param perturbProb The probability to perturb a genome's weight when we decide
   *        to mutate it. The probability of apply unbiased weight mutation is 1 - perturbProb.
   */
  void MutateWeight(Genome& genome, double mutateProb, double perturbProb, double mutateSize) 
  {
    double p = mlpack::math::Random();  // rand 0~1
    if (p > mutateProb) return;
    
    for (int i=0; i<genome.aLinkGenes.size(); ++i)
    {  
      double p2 = mlpack::math::Random();
      if (p2 < perturbProb)
      {  // Biased weight mutation.
        double deltaW = mlpack::math::RandNormal(0, mutateSize);
        double oldW = genome.aLinkGenes[i].Weight();
        genome.aLinkGenes[i].Weight(oldW + deltaW);
      } else
      {  // Unbiased weight mutation.
        double weight = mlpack::math::RandNormal(0, mutateSize);
        genome.aLinkGenes[i].Weight(weight);
      }
    }
  }

  /**
   * Compare which genome is better.
   *
   * Fitness smaller is better. When fitness is the same, the genome with
   * simpler structure (less link) is better. If genome lg better than rg,
   * return true, otherwise return false.
   *
   * @param lg One of the genome to compare.
   * @param rg Another genome to compare.
   */
  static bool CompareGenome(const Genome& lg, const Genome& rg)
  {
    assert(lg.Fitness() != DBL_MAX);
    assert(rg.Fitness() != DBL_MAX);

    if (lg.Fitness() < rg.Fitness())
    {
      return true;
    } else if (rg.Fitness() < lg.Fitness())
    {
      return false;
    } else if (lg.NumLink() < rg.NumLink())
    {
      return true;
    } else if (rg.NumLink() < lg.NumLink())
    {
      return false;
    } else if (mlpack::math::Random() < 0.5)
    {
      return true;
    } else
    {
      return false;
    }
  }

  /**
   * Crossover link weights. Assume momGenome is the better genome, childGenome is empty.
   *
   * @param momGenome The mom genome of crossover operation.
   * @param dadGenome The dad genome of crossover operation.
   * @param childGenome The new genome generated by crossover will be saved by it.
   */
  void CrossoverLinkAndNeuron(Genome& momGenome, Genome& dadGenome, Genome& childGenome)
  {
    childGenome.NumInput(momGenome.NumInput());
    childGenome.NumOutput(momGenome.NumOutput());

    // Add input and output neuron genes to child genome.
    for (int i=0; i<(momGenome.NumInput() + momGenome.NumOutput()); ++i)
    {
      childGenome.aNeuronGenes.push_back(momGenome.aNeuronGenes[i]);
    }

    // Iterate to add link genes and neuron genes to child genome.
    for (int i=0; i<momGenome.NumLink(); ++i)
    {
      int innovId = momGenome.aLinkGenes[i].InnovationId();      
      int idx = dadGenome.GetLinkIndex(innovId);
      bool linkContainedInDad = (idx != -1);
      double randNum = mlpack::math::Random();

      // Exceed or disjoint link, add to child.
      if (!linkContainedInDad)
      {  
        childGenome.AddLink(momGenome.aLinkGenes[i]);

        // Add from neuron.
        int idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        int idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }
        continue;
      }

      // Common link in both parents, add mom's to child with probability 0.5.
      if (linkContainedInDad && randNum < 0.5)
      {
        childGenome.AddLink(momGenome.aLinkGenes[i]);

        // Add from neuron.
        int idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        int idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].FromNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.aLinkGenes[i].ToNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.aNeuronGenes[idxInParent]);
        }
        continue;
      }

      // Common link in both parents, add dad's to child with probability 0.5.
      if (linkContainedInDad && randNum >= 0.5)
      {
        childGenome.AddLink(dadGenome.aLinkGenes[idx]);

        // Add from neuron.
        int idxInChild = childGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].FromNeuronId());
        int idxInParent = dadGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].FromNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(dadGenome.aNeuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].ToNeuronId());
        idxInParent = dadGenome.GetNeuronIndex(dadGenome.aLinkGenes[idx].ToNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(dadGenome.aNeuronGenes[idxInParent]);
        }
        continue;
      }  
    }
  }

  /**
   * Crossover two genome to get one genome.
   *
   * It is used to wrap the CrossoverLinkAndNeuron function. So that we
   * don't to worry about which parent is the better genome.
   *
   * @param genome1 One of the parent genome for crossover.
   * @param genome2 Another parent genome for crossover.
   * @param childGenome The child genome generated by crossover.
   */
  void Crossover(Genome& genome1, Genome& genome2, Genome& childGenome)
  {
    if (CompareGenome(genome1, genome2))
    {  // genome1 is better
      CrossoverLinkAndNeuron(genome1, genome2, childGenome);
    } else
    {  // genome2 is better
      CrossoverLinkAndNeuron(genome2, genome1, childGenome);
    }
  }

  /**
   * Measure two genomes' disjoint (including exceed).
   *
   * NOTICE: we can separate into disjoint and exceed. But currently maybe it is enough.
   *
   * @param genome1 One genome.
   * @param genome2 Another genome. Compare genome 1 and genome 2's disjoint.
   */
  double Disjoint(Genome& genome1, Genome& genome2)
  {
    double numDisjoint = 0;

    for (int i=0; i<genome1.NumLink(); ++i)
    {
      int innovId = genome1.aLinkGenes[i].InnovationId();
      bool linkContainedInGenome2 = genome2.ContainLink(innovId);
      if (!linkContainedInGenome2)
      {
        ++numDisjoint;
      } 
    }

    for (int i=0; i<genome2.NumLink(); ++i)
    {
      int innovId = genome2.aLinkGenes[i].InnovationId();
      bool linkContainedInGenome1 = genome1.ContainLink(innovId);
      if (!linkContainedInGenome1)
      {
        ++numDisjoint;
      }
    }

    int largerGenomeSize = std::max(genome1.NumLink(), genome2.NumLink());
    double deltaD = numDisjoint / largerGenomeSize;
    return deltaD; 
  }

  /**
   * Measure two genomes' weight difference.
   *
   * @param genome1 One genome.
   * @param genome2 Another genome. Compare genome 1 and genome 2's weight difference.
   */
  double WeightDiff(Genome& genome1, Genome& genome2)
  {
    double deltaW = 0;
    int coincident = 0;

    for (int i=0; i<genome1.NumLink(); ++i)
    {
      int linkEnabledInGenome1 = (int) genome1.aLinkGenes[i].Enabled();
      int innovId = genome1.aLinkGenes[i].InnovationId();
      int idx = genome2.GetLinkIndex(innovId);
      bool linkContainedInGenome2 = (idx != -1);

      if (linkContainedInGenome2)
      {
        int linkEnabledInGenome2 = (int) genome2.aLinkGenes[idx].Enabled();
        deltaW += std::abs(genome1.aLinkGenes[i].Weight() * linkEnabledInGenome1 - 
                           genome2.aLinkGenes[idx].Weight() * linkEnabledInGenome2);
        ++coincident;
      }
    }

    deltaW = deltaW / coincident;
    return deltaW;
  }

  /**
   * Whether two genome belong to same species or not.
   *
   * @param genome1 One genome.
   * @param genome2 Another genome. Judge whether genome1 and genome2 are same species.
   */
  bool IsSameSpecies(Genome& genome1, Genome& genome2)
  {
    double deltaD = Disjoint(genome1, genome2);
    double deltaW = WeightDiff(genome1, genome2);
    double delta = aCoeffDisjoint * deltaD + aCoeffWeightDiff * deltaW;

    if (delta < aCompatThreshold)
    {
      return true;
    } 
    
    return false;
  }

  /**
   * Add genome to existing a population's existing species or create new species.
   *
   * @param population Add genome to population.
   * @param genome The genome to add.
   */
  void AddGenomeToSpecies(Population& population, Genome& genome)
  {
    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      if (population.aSpecies[i].aGenomes.size() > 0)
      {
        if (IsSameSpecies(population.aSpecies[i].aGenomes[0], genome))
        {  // each first genome in species is the representative genome.
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

  /**
   * Remove stale species.
   *
   * If a species has been evolved for over than aStaleAgeThreshold generations,
   * and its best genome's fitness doesn't change, then it is a stale species.
   * It will be removed from population.
   */
  void RemoveStaleSpecies(Population& population)
  {
    for (std::vector<Species>::iterator it = population.aSpecies.begin();
         it != population.aSpecies.end();  /*it++*/)
    {
      if(it->StaleAge() > aStaleAgeThreshold)
      {
        it = population.aSpecies.erase(it);
      }else
      {
        ++it;
      }
    }
  }

  /**
   * Aggregate population's genomes.
   *
   * Put all the genomes in a population into a genome vector.
   *
   * @param population We will aggregate its genomes.
   * @param genomes The genome vector to save population's genomes.
   */
  void AggregateGenomes(Population& population, std::vector<Genome>& genomes)
  {
    genomes.clear();
    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      for (int j=0; j<population.aSpecies[i].aGenomes.size(); ++j)
      {
        genomes.push_back(population.aSpecies[i].aGenomes[j]);
      }
    }
  }

  /**
   * Sort genomes by fitness. Smaller fitness is better and put first.
   *
   * @param genomes The genome vector to sort.
   */
  void SortGenomes(std::vector<Genome>& genomes) {
    std::sort(genomes.begin(), genomes.end(), Species::CompareGenome);
  }

  /**
   * Get genome index in a genome vector. If not found, return -1.
   *
   * @param genomes The genome vector we are retrieval.
   * @param id The genome id we are searching in the genomes vector.
   */
  int GetGenomeIndex(std::vector<Genome>& genomes, int id)
  {
    for (int i=0; i<genomes.size(); ++i)
    {
      if (genomes[i].Id() == id)
        return i;
    }
    return -1;
  }

  /**
   * Calculate species' average rank in population by fitness. Bigger is better.
   *
   * A species' average rank is the average of all its genomes. Besides, it is reversed to
   * make sure a bigger rank is better. For example, if the population contains N genomes,
   * the best genome's rank is N, and the worst is 1.
   *
   * @param population The population we are focusing on.
   * @param speciesAverageRank A vector to save the average rank of each species in population.
   */
  void CalcSpeciesAverageRank(Population& population, std::vector<double>& speciesAverageRank) 
  {
    std::vector<Genome> genomes;
    AggregateGenomes(population, genomes);
    SortGenomes(genomes);
    speciesAverageRank.clear();

    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      double averageRank = 0;
      int speciesSize = population.aSpecies[i].aGenomes.size();
      for (int j=0; j<speciesSize; ++j)
      {
        averageRank += genomes.size() - GetGenomeIndex(genomes, population.aSpecies[i].aGenomes[j].Id());
      }
      averageRank = averageRank / speciesSize;
      speciesAverageRank.push_back(averageRank);
    }
  }

  /**
   * Remove weak species.
   *
   * A species is considered weak if its average rank is lower than 
   * the average of all species.
   *
   * @param population The population we are focusing on.
   */
  void RemoveWeakSpecies(Population& population)
  {
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(population, speciesAverageRank);
    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);

    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      double weak = (std::floor(speciesAverageRank[i] * population.NumSpecies() / totalAverageRank)
                     < 1);
      if (weak)
      {
        population.RemoveSpecies(i);
      }
    }
  }

  /**
   * Remove empty species.
   *
   * @param population The population we are focusing on.
   */
  void RemoveEmptySpecies(Population& population)
  {
    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      if (population.aSpecies[i].aGenomes.size() == 0)
      {
        population.aSpecies.erase(population.aSpecies.begin() + i);
      }
    }
  }

  /**
   * Remove a portion of weak genomes in each species.
   *
   * @param population The population we are focusing on.
   * @param percentageToRemove The percentage of genomes that 
   *        will be removed from each species.
   */
  void CullSpecies(Population& population, double percentageToRemove)
  {
    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      population.aSpecies[i].SortGenomes();
      int numRemove = std::floor(population.aSpecies[i].aGenomes.size() * percentageToRemove);
      while (numRemove > 0)
      {
        population.aSpecies[i].aGenomes.pop_back();
        --numRemove;
      }
    }
    RemoveEmptySpecies(population);
  }

  /**
   * Only keep the best genome in each species.
   *
   * @param population The population we are focusing on.
   */
  void CullSpeciesToOne(Population& population)
  {
    for (int i=0; i<population.aSpecies.size(); ++i)
    {
      population.aSpecies[i].SortGenomes();
      int speciesSize = population.aSpecies[i].aGenomes.size();
      if (speciesSize > 0)
      {
        Genome bestGenome = population.aSpecies[i].aGenomes[0];
        population.aSpecies[i].aGenomes.clear();
        population.aSpecies[i].aGenomes.push_back(bestGenome);
      }
    }
    RemoveEmptySpecies(population);
  }

  /**
   * Mutate a genome by combining different mutations.
   *
   * NOTICE: how we organize different mutations is kind of flexible.
   *
   * @param genome The genome we are focusing on to apply mutation operations.
   */
  void Mutate(Genome& genome)
  {
    // Mutate weights.
    MutateWeight(genome, aMutateWeightProb, aPerturbWeightProb, aMutateWeightSize);

    // Mutate add forward link.
    double p = aMutateAddForwardLinkProb;
    while (p > 0)
    {  // so p can be bigger than 1 and mutate can happen multiple times.
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, FORWARD_LINK, aMutateAddForwardLinkProb);
      }
      --p;
    }

    // Mutate add backward link.
    p = aMutateAddBackwardLinkProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, BACKWARD_LINK, aMutateAddBackwardLinkProb);
      }
      --p;
    }

    // Mutate add recurrent link.
    p = aMutateAddRecurrentLinkProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, RECURRENT_LINK, aMutateAddRecurrentLinkProb);
      }
      --p;
    }

    // Mutate add bias link.
    p = aMutateAddBiasLinkProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, BIAS_LINK, aMutateAddBiasLinkProb);
      }
      --p;
    }

    // Mutate add neuron.
    p = aMutateAddNeuronProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddNeuron(genome, aMutateAddNeuronProb, false);
      }
      --p;
    }

    // Mutate enabled node to disabled.
    p = aMutateEnabledProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateEnableDisable(genome, true, aMutateEnabledProb);
      }
      --p;
    }

    // Mutate disabled node to enabled.
    p = aMutateDisabledProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateEnableDisable(genome, false, aMutateDisabledProb);
      }
      --p;
    }
  }

  /**
   * Breed child for a species.
   *
   * Crossover to born a child, or copy a child, then mutate it.
   * Return true if a child genome is successfully born.
   * NOTICE: can have different ways to breed a child. It is flexible.
   *
   * @param species Breed child for this species.
   * @param childGenome If new child genome born, it will be saved by this childGenome.
   * @param crossoverProb The crossover probability to generate child genome.
   */
  bool BreedChild(Species& species, Genome& childGenome, double crossoverProb)
  {
    double p = mlpack::math::Random();
    int speciesSize = species.aGenomes.size();

    if (speciesSize == 0)
      return false;

    if (p < crossoverProb)
    {
      int idx1 = mlpack::math::RandInt(0, speciesSize);
      int idx2 = mlpack::math::RandInt(0, speciesSize);
      if (idx1 != idx2)
      {
        Crossover(species.aGenomes[idx1], species.aGenomes[idx2], childGenome);
      } else
      {
        return false;
      }
    } else
    {
      int idx = mlpack::math::RandInt(0, speciesSize);
      childGenome = species.aGenomes[idx];
    }

    Mutate(childGenome);

    return true;
  }

  /**
   * Initialize population.
   *
   * Create a bunch of genomes using the seed genome to construct a population.
   */
  void InitPopulation()
  {
    aPopulation = Population(aSeedGenome, aPopulationSize);
  }

  /**
   * Reproduce next generation of population. Key function of NEAT algorithm.
   *
   * NOTICE: steps in reproduce are also kind of flexible.
   */
  void Reproduce()
  {
    // keep previous best genome.
    std::vector<Genome> childGenomes;
    Genome lastBestGenome = aPopulation.BestGenome();
    childGenomes.push_back(lastBestGenome);

    // Remove weak genomes in each species.
    CullSpecies(aPopulation, aCullSpeciesPercentage);

    // Remove stale species, weak species.
    if (aPopulation.aSpecies.size() > aNumSpeciesThreshold)
    {
      RemoveStaleSpecies(aPopulation);
      RemoveWeakSpecies(aPopulation);
    }

    // Breed children in each species. 
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(aPopulation, speciesAverageRank);
    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);

    for (int i=0; i<aPopulation.aSpecies.size(); ++i)
    {
      int numBreed = std::floor(speciesAverageRank[i] * aPopulationSize / totalAverageRank) - 1;
      int numBreedSuccess = 0;

      while (numBreedSuccess < numBreed)
      {
        Genome genome;
        bool hasBaby = BreedChild(aPopulation.aSpecies[i], genome, aCrossoverRate);
        if (hasBaby)
        {
          childGenomes.push_back(genome);
          ++numBreedSuccess;
        }
      }
    }

    // Keep the best in each species.
    CullSpeciesToOne(aPopulation);

    // Random choose species and breed child until reach population size.
    int currentNumGenome = childGenomes.size() + aPopulation.PopulationSize();
    while (currentNumGenome < aPopulationSize)
    {
      int speciesIndex = mlpack::math::RandInt(0, aPopulation.aSpecies.size());
      Genome genome;
      bool hasBaby = BreedChild(aPopulation.aSpecies[speciesIndex], genome, aCrossoverRate);
      if (hasBaby)
      {
        childGenomes.push_back(genome);
        ++currentNumGenome;
      }
    }

    // Speciate genomes into new species.
    for (int i=0; i<childGenomes.size(); ++i)
    {
      AddGenomeToSpecies(aPopulation, childGenomes[i]);
    }

    //DEBUGGING!!!!!!!!!
    printf("Species sizes are: ");
    for (int s=0; s<aPopulation.aSpecies.size(); ++s)
    {
      std::cout<< aPopulation.aSpecies[s].aGenomes.size() << "  ";
    }
    printf("\n");
    //DEBUGGING!!!!!!!!!

    // Reassign genome IDs.
    aPopulation.ReassignGenomeId();
  }

  /**
   * Evaluate genomes in population.
   * Set genomes' fitness, species' and population's best fitness and genome.
   */
  void Evaluate()
  {
    for (int i=0; i<aPopulation.aSpecies.size(); ++i)
    {
      for (int j=0; j<aPopulation.aSpecies[i].aGenomes.size(); ++j)
      {
        aPopulation.aSpecies[i].aGenomes[j].Flush();
        double fitness = aTask.EvalFitness(aPopulation.aSpecies[i].aGenomes[j]);
        aPopulation.aSpecies[i].aGenomes[j].Fitness(fitness);
      }

      double oldSpeciesBestFitness = aPopulation.aSpecies[i].BestFitness();
      aPopulation.aSpecies[i].SetBestFitnessAndGenome();
      double newSpeciesBestFitness = aPopulation.aSpecies[i].BestFitness();
      if (newSpeciesBestFitness < oldSpeciesBestFitness)
      {
        aPopulation.aSpecies[i].StaleAge(0);
      } else
      {
        int staleAge = aPopulation.aSpecies[i].StaleAge();
        aPopulation.aSpecies[i].StaleAge(staleAge + 1);
      }
    }
    aPopulation.SetBestFitnessAndGenome();
  }

  /**
   * Evolve population of genomes to get a task's solution genome.
   *
   * This function is the whole progress of NEAT algorithm.
   */
  bool Evolve()
  {
    // Generate initial species at random.
    int generation = 0;
    InitPopulation();

    // Speciate genomes into species.
    std::vector<Genome> genomes;
    AggregateGenomes(aPopulation, genomes);
    aPopulation.aSpecies.clear();
    for (int i=0; i<genomes.size(); ++i)
    {
      AddGenomeToSpecies(aPopulation, genomes[i]);
    }
    
    // Repeat
    while (generation < aMaxGeneration)
    {
      // Evaluate all genomes in population.
      Evaluate();

      // Output some information.
      printf("Generation: %zu\tBest fitness: %f\n", generation, aPopulation.BestFitness());
      //Log::Info << "Generation: " << generation << " best fitness: " <<  aPopulation.BestFitness() << std::endl;
      if (aTask.Success())
      {
        printf("Task succeed in %zu iterations.\n", generation);
        return true;
      }

      // Reproduce next generation.
      Reproduce();
      ++generation;
    }

    return false;
  }
  
 private:

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEAT_HPP
