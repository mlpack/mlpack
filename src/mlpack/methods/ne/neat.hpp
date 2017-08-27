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

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(fromNeuronId, "fromNeuronId");
    ar & data::CreateNVP(toNeuronId, "toNeuronId");
    ar & data::CreateNVP(newLinkInnovId, "newLinkInnovId");
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

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(splitLinkInnovId, "splitLinkInnovId");
    ar & data::CreateNVP(actFuncType, "actFuncType");
    ar & data::CreateNVP(newNeuronId, "newNeuronId");
    ar & data::CreateNVP(newInputLinkInnovId, "newInputLinkInnovId");
    ar & data::CreateNVP(newOutputLinkInnovId, "newOutputLinkInnovId");
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
class NEAT
{
 public:

  //! Seed genome. It is used for initialize population.
  Genome seedGenome;

  //! Population to evolve.
  Population population;

  //! Population size.
  int populationSize;

  //! List of link innovations.
  std::vector<LinkInnovation> linkInnovations;

  //! List of neuron innovations.
  std::vector<NeuronInnovation> neuronInnovations;

  //! Next neuron id.
  int nextNeuronId;

  //! Next link id.
  int nextLinkInnovId;

  //! Max number of generation to evolve.
  int maxGeneration;

  //! Efficient for disjoint.
  double coeffDisjoint;

  //! Efficient for weight difference.
  double coeffWeightDiff;

  //! Threshold for judge whether belong to same species.
  double compatThreshold;

  //! Threshold for species stale age.
  int staleAgeThreshold;

  //! Crossover rate.
  double crossoverRate;

  //! Percentage to remove in each species.
  double cullSpeciesPercentage;

  //! Probability to mutate a genome's weight.
  double mutateWeightProb;

  //! Probability to mutate a genome's weight in biased way (add Gaussian perturb noise).
  double perturbWeightProb;

  //! The Gaussian noise variance when mutating genome weights.
  double mutateWeightSize;

  //! Probability to add a forward link.
  double mutateAddForwardLinkProb;

  //! Probability to add a backward link.
  double mutateAddBackwardLinkProb;

  //! Probability to add a recurrent link.
  double mutateAddRecurrentLinkProb;

  //! Probability to add a bias link.
  double mutateAddBiasLinkProb;

  //! Probability to add neuron to genome.
  double mutateAddNeuronProb;

  //! Probability to turn enabled link to disabled.
  double mutateEnabledProb;

  //! Probability to turn disabled link to enabled.
  double mutateDisabledProb;

  //! Species number threshold.
  int numSpeciesThreshold;

  //! Whether the activation function of new neuron is random or not.
  bool randomTypeNewNeuron;

   NEAT()
  {
  }

  // /**
  //  * Default constructor.
  //  */
  // NEAT()
  // {
  //   mlpack::math::RandomSeed(1);

  //   // Set NEAT algorithm parameters (except task).
  //   populationSize = 500;
  //   maxGeneration = 500;
  //   coeffDisjoint = 2.0;
  //   coeffWeightDiff = 0.4;
  //   compatThreshold = 1.0;
  //   staleAgeThreshold = 15;
  //   crossoverRate = 0.75;
  //   cullSpeciesPercentage = 0.5;
  //   mutateWeightProb = 0.2;
  //   perturbWeightProb = 0.9;
  //   mutateWeightSize = 0.1;
  //   mutateAddForwardLinkProb = 0.9;
  //   mutateAddBackwardLinkProb = 0;
  //   mutateAddRecurrentLinkProb = 0;
  //   mutateAddBiasLinkProb = 0;
  //   mutateAddNeuronProb = 0.6;
  //   mutateEnabledProb = 0.2;
  //   mutateDisabledProb = 0.2;
  //   numSpeciesThreshold = 10;
  //   randomTypeNewNeuron = false;

  //   // Construct seed genome for xor task.
  //   int id = 0;
  //   int numInput = 3;
  //   int numOutput = 1;
  //   double fitness = -1;
  //   std::vector<NeuronGene> neuronGenes;
  //   std::vector<LinkGene> linkGenes;

  //   NeuronGene inputGene1(0, INPUT, LINEAR, 0, std::vector<double>(), 0, 0);
  //   NeuronGene inputGene2(1, INPUT, LINEAR, 0, std::vector<double>(), 0, 0);
  //   NeuronGene biasGene(2, BIAS, LINEAR, 0, std::vector<double>(), 0, 0);
  //   NeuronGene outputGene(3, OUTPUT, SIGMOID, 1, std::vector<double>(), 0, 0);
  //   NeuronGene hiddenGene(4, HIDDEN, SIGMOID, 0.5, std::vector<double>(), 0, 0);

  //   neuronGenes.push_back(inputGene1);
  //   neuronGenes.push_back(inputGene2);
  //   neuronGenes.push_back(biasGene);
  //   neuronGenes.push_back(outputGene);
  //   neuronGenes.push_back(hiddenGene);

  //   LinkGene link1(0, 3, 0, 0, true);
  //   LinkGene link2(1, 3, 1, 0, true);
  //   LinkGene link3(2, 3, 2, 0, true);
  //   LinkGene link4(0, 4, 3, 0, true);
  //   LinkGene link5(1, 4, 4, 0, true);
  //   LinkGene link6(2, 4, 5, 0, true);
  //   LinkGene link7(4, 3, 6, 0, true);

  //   linkGenes.push_back(link1);
  //   linkGenes.push_back(link2);
  //   linkGenes.push_back(link3);
  //   linkGenes.push_back(link4);
  //   linkGenes.push_back(link5);
  //   linkGenes.push_back(link6);
  //   linkGenes.push_back(link7);

  //   Genome seedGenome = Genome(0, 
  //                              neuronGenes,
  //                              linkGenes,
  //                              numInput,
  //                              numOutput,
  //                              fitness);

  //   // Set neat members. (Except task.)
  //   this->seedGenome = seedGenome;
  //   nextNeuronId = seedGenome.NumNeuron();
  //   nextLinkInnovId = seedGenome.NumLink();


  //   // Generate initial species at random.
  //   int generation = 0;
  //   InitPopulation();

  //   // Speciate genomes into species.
  //   std::vector<Genome> genomes;
  //   AggregateGenomes(population, genomes);
  //   population.species.clear();
  //   for (int i = 0; i < genomes.size(); ++i)
  //   {
  //     AddGenomeToSpecies(population, genomes[i]);
  //   }
  // }

  /**
   * Parametric constructor.
   *
   * @param task The task to solve.
   * @param seedGenome The genome to initialize population.
   * @param params The Parameter object that contains algorithm parameters.
   */
  NEAT(Genome& seedGenome, Parameters& params)
  {
    this->seedGenome = seedGenome;
    nextNeuronId = seedGenome.NumNeuron();
    nextLinkInnovId = seedGenome.NumLink();
    populationSize = params.populationSize;
    maxGeneration = params.maxGeneration;
    coeffDisjoint = params.coeffDisjoint;
    coeffWeightDiff = params.coeffWeightDiff;
    compatThreshold = params.compatThreshold;
    staleAgeThreshold = params.staleAgeThreshold;
    crossoverRate = params.crossoverRate;
    cullSpeciesPercentage = params.cullSpeciesPercentage;
    mutateWeightProb = params.mutateWeightProb;
    perturbWeightProb = params.perturbWeightProb;
    mutateWeightSize = params.mutateWeightSize;
    mutateAddForwardLinkProb = params.mutateAddForwardLinkProb;
    mutateAddBackwardLinkProb = params.mutateAddBackwardLinkProb;
    mutateAddRecurrentLinkProb = params.mutateAddRecurrentLinkProb;
    mutateAddBiasLinkProb = params.mutateAddBiasLinkProb;
    mutateAddNeuronProb = params.mutateAddNeuronProb;
    mutateEnabledProb = params.mutateEnabledProb;
    mutateDisabledProb = params.mutateDisabledProb;
    numSpeciesThreshold = params.numSpeciesThreshold;
    randomTypeNewNeuron = params.randomTypeNewNeuron;


    // Generate initial species at random.
    int generation = 0;
    InitPopulation();

    // Speciate genomes into species.
    std::vector<Genome> genomes;
    AggregateGenomes(population, genomes);
    population.species.clear();
    for (int i = 0; i < genomes.size(); ++i)
    {
      AddGenomeToSpecies(population, genomes[i]);
    }
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
    for (int i = 0; i < linkInnovations.size(); ++i)
    {
      if (linkInnovations[i].fromNeuronId == fromNeuronId && 
          linkInnovations[i].toNeuronId == toNeuronId)
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
    for (int i = 0; i < neuronInnovations.size(); ++i)
    {
      if (neuronInnovations[i].splitLinkInnovId == splitLinkInnovId &&
          neuronInnovations[i].actFuncType == actFuncType)
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
    linkInnov.newLinkInnovId = nextLinkInnovId++;
    linkInnovations.push_back(linkInnov);
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
    neuronInnov.newNeuronId = nextNeuronId++;
    neuronInnov.newInputLinkInnovId = nextLinkInnovId++;
    neuronInnov.newOutputLinkInnovId = nextLinkInnovId++;
    neuronInnovations.push_back(neuronInnov);
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
    for (int i = 0; i < genome.NumLink(); ++i)
    {
      if (genome.linkGenes[i].FromNeuronId() == fromNeuronId &&
          genome.linkGenes[i].ToNeuronId() == toNeuronId)
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

    if (genome.neuronGenes.size() == 0) return;

    // Select from neuron and to neuron.
    int fromNeuronIdx = -1;
    int fromNeuronId = -1;
    int toNeuronIdx = -1;
    int toNeuronId = -1;

    switch (linkType)
    {
      case FORWARD_LINK:
        // Select from neuron.
        fromNeuronIdx = mlpack::math::RandInt(0, genome.neuronGenes.size());
        fromNeuronId = genome.neuronGenes[fromNeuronIdx].Id();

        // Select to neuron which cannot be input.
        toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.neuronGenes.size()); 
        toNeuronId = genome.neuronGenes[toNeuronIdx].Id();

        // Don't allow same depth connection.
        if (genome.neuronGenes[fromNeuronIdx].Depth() == genome.neuronGenes[toNeuronIdx].Depth())
        {
          return;
        }

        // Swap if backward.
        if (genome.neuronGenes[fromNeuronIdx].Depth() > genome.neuronGenes[toNeuronIdx].Depth())
        {
          std::swap(fromNeuronIdx, toNeuronIdx);
          std::swap(fromNeuronId, toNeuronId);
        }

        break;
      case BACKWARD_LINK:
        // Select from neuron.
        fromNeuronIdx = mlpack::math::RandInt(0, genome.neuronGenes.size());
        fromNeuronId = genome.neuronGenes[fromNeuronIdx].Id();

        // Select to neuron which cannot be input.
        toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.neuronGenes.size()); 
        toNeuronId = genome.neuronGenes[toNeuronIdx].Id();

        // Don't allow same depth connection.
        if (genome.neuronGenes[fromNeuronIdx].Depth() == genome.neuronGenes[toNeuronIdx].Depth())
        {
          return;
        }

        // Swap if forward.
        if (genome.neuronGenes[fromNeuronIdx].Depth() < genome.neuronGenes[toNeuronIdx].Depth()) 
        {
          std::swap(fromNeuronIdx, toNeuronIdx);
          std::swap(fromNeuronId, toNeuronId);
        }

        break;
      case RECURRENT_LINK:
        // Select recurrent neuron.
        fromNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.neuronGenes.size());
        fromNeuronId = genome.neuronGenes[fromNeuronIdx].Id();
        toNeuronIdx = fromNeuronIdx;
        toNeuronId = fromNeuronId;
        break;
      case BIAS_LINK:
        // Set from neuron as the BIAS neuron.
        fromNeuronIdx = genome.NumInput() - 1;
        fromNeuronId = genome.neuronGenes[fromNeuronIdx].Id();

        // Select to neuron which cannot be input.
        toNeuronIdx = mlpack::math::RandInt(genome.NumInput(), genome.neuronGenes.size()); 
        toNeuronId = genome.neuronGenes[toNeuronIdx].Id();
        break;
      default:
        return;
    }

    // Check link already exist or not.
    int linkIdx = IsLinkExist(genome, fromNeuronId, toNeuronId);
    if (linkIdx != -1)
    {
      genome.linkGenes[linkIdx].Enabled(true);
      return;
    }

    // Check innovation already exist or not.
    int innovIdx = CheckLinkInnovation(fromNeuronId, toNeuronId);
    if (innovIdx != -1)
    {
      LinkGene linkGene(fromNeuronId,
                        toNeuronId,
                        linkInnovations[innovIdx].newLinkInnovId,
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
    if (!genome.linkGenes[linkIdx].Enabled()) return;

    genome.linkGenes[linkIdx].Enabled(false);
    NeuronGene fromNeuron;
    genome.GetNeuronById(genome.linkGenes[linkIdx].FromNeuronId(), fromNeuron);
    NeuronGene toNeuron;
    genome.GetNeuronById(genome.linkGenes[linkIdx].ToNeuronId(), toNeuron);

    // Check innovation already exist or not.
    int splitLinkInnovId = genome.linkGenes[linkIdx].InnovationId();
    ActivationFuncType actFuncType = SIGMOID;
    if (randomActFuncType) 
    {
      actFuncType = static_cast<ActivationFuncType>(std::rand() % ActivationFuncType::COUNT);  // TODO: use discrete distribution.
    }
    int innovIdx = CheckNeuronInnovation(splitLinkInnovId, actFuncType);

    // If existing innovation.
    if (innovIdx != -1)
    {
      // Check whether this genome already contains the neuron.
      int neuronIdx = genome.GetNeuronIndex(neuronInnovations[innovIdx].newNeuronId);
      if (neuronIdx != -1)  // The neuron already exist.
      {
        // Enable the input and output link.
        int inputLinkIdx = genome.GetLinkIndex(neuronInnovations[innovIdx].newInputLinkInnovId);
        int outputLinkIdx = genome.GetLinkIndex(neuronInnovations[innovIdx].newOutputLinkInnovId);
        genome.linkGenes[inputLinkIdx].Enabled(true);
        genome.linkGenes[outputLinkIdx].Enabled(true);
      } else
      {
        NeuronGene neuronGene(neuronInnovations[innovIdx].newNeuronId,
                              HIDDEN,
                              actFuncType,
                              (fromNeuron.Depth() + toNeuron.Depth()) / 2,
                              std::vector<double>(),
                              0,
                              0);
        LinkGene inputLink(genome.linkGenes[linkIdx].FromNeuronId(),
                           neuronInnovations[innovIdx].newNeuronId,
                           neuronInnovations[innovIdx].newInputLinkInnovId,
                           1,
                           true);
        LinkGene outputLink(neuronInnovations[innovIdx].newNeuronId,
                            genome.linkGenes[linkIdx].ToNeuronId(),
                            neuronInnovations[innovIdx].newOutputLinkInnovId,
                            genome.linkGenes[linkIdx].Weight(),
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
    inputLinkInnov.fromNeuronId = genome.linkGenes[linkIdx].FromNeuronId();
    inputLinkInnov.toNeuronId = neuronInnov.newNeuronId;
    inputLinkInnov.newLinkInnovId = neuronInnov.newInputLinkInnovId;
    linkInnovations.push_back(inputLinkInnov);

    LinkInnovation outputLinkInnov;
    outputLinkInnov.fromNeuronId = neuronInnov.newNeuronId;
    outputLinkInnov.toNeuronId = genome.linkGenes[linkIdx].ToNeuronId();
    outputLinkInnov.newLinkInnovId = neuronInnov.newOutputLinkInnovId;
    linkInnovations.push_back(outputLinkInnov);
    
    // Add neuron, input link, output link.
    NeuronGene neuronGene(neuronInnov.newNeuronId,
                          HIDDEN,
                          actFuncType,
                          (fromNeuron.Depth() + toNeuron.Depth()) / 2,
                          std::vector<double>(),
                          0,
                          0);
    LinkGene inputLink(genome.linkGenes[linkIdx].FromNeuronId(),
                       neuronInnov.newNeuronId,
                       neuronInnov.newInputLinkInnovId,
                       1,
                       true);
    LinkGene outputLink(neuronInnov.newNeuronId,
                        genome.linkGenes[linkIdx].ToNeuronId(),
                        neuronInnov.newOutputLinkInnovId,
                        genome.linkGenes[linkIdx].Weight(),
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
    for (int i = 0; i < genome.NumLink(); ++i)
    {
      if (genome.linkGenes[i].Enabled() == enabled)
      {
        linkIndexs.push_back(i);
      }
    }
    
    if (linkIndexs.size()>0)
    {
      int idx = linkIndexs[mlpack::math::RandInt(0, linkIndexs.size())];
      genome.linkGenes[idx].Enabled(!enabled);
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
    
    for (int i = 0; i < genome.linkGenes.size(); ++i)
    {  
      double p2 = mlpack::math::Random();
      if (p2 < perturbProb)
      {  // Biased weight mutation.
        double deltaW = mlpack::math::RandNormal(0, mutateSize);
        double oldW = genome.linkGenes[i].Weight();
        genome.linkGenes[i].Weight(oldW + deltaW);
      } else
      {  // Unbiased weight mutation.
        double weight = mlpack::math::RandNormal(0, mutateSize);
        genome.linkGenes[i].Weight(weight);
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
    for (int i = 0; i < (momGenome.NumInput() + momGenome.NumOutput()); ++i)
    {
      childGenome.neuronGenes.push_back(momGenome.neuronGenes[i]);
    }

    // Iterate to add link genes and neuron genes to child genome.
    for (int i = 0; i < momGenome.NumLink(); ++i)
    {
      int innovId = momGenome.linkGenes[i].InnovationId();      
      int idx = dadGenome.GetLinkIndex(innovId);
      bool linkContainedInDad = (idx != -1);
      double randNum = mlpack::math::Random();

      // Exceed or disjoint link, add to child.
      if (!linkContainedInDad)
      {  
        childGenome.AddLink(momGenome.linkGenes[i]);

        // Add from neuron.
        int idxInChild = childGenome.GetNeuronIndex(momGenome.linkGenes[i].FromNeuronId());
        int idxInParent = momGenome.GetNeuronIndex(momGenome.linkGenes[i].FromNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.neuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(momGenome.linkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.linkGenes[i].ToNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.neuronGenes[idxInParent]);
        }
        continue;
      }

      // Common link in both parents, add mom's to child with probability 0.5.
      if (linkContainedInDad && randNum < 0.5)
      {
        childGenome.AddLink(momGenome.linkGenes[i]);

        // Add from neuron.
        int idxInChild = childGenome.GetNeuronIndex(momGenome.linkGenes[i].FromNeuronId());
        int idxInParent = momGenome.GetNeuronIndex(momGenome.linkGenes[i].FromNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.neuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(momGenome.linkGenes[i].ToNeuronId());
        idxInParent = momGenome.GetNeuronIndex(momGenome.linkGenes[i].ToNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(momGenome.neuronGenes[idxInParent]);
        }
        continue;
      }

      // Common link in both parents, add dad's to child with probability 0.5.
      if (linkContainedInDad && randNum >= 0.5)
      {
        childGenome.AddLink(dadGenome.linkGenes[idx]);

        // Add from neuron.
        int idxInChild = childGenome.GetNeuronIndex(dadGenome.linkGenes[idx].FromNeuronId());
        int idxInParent = dadGenome.GetNeuronIndex(dadGenome.linkGenes[idx].FromNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(dadGenome.neuronGenes[idxInParent]);
        }

        // Add to neuron.
        idxInChild = childGenome.GetNeuronIndex(dadGenome.linkGenes[idx].ToNeuronId());
        idxInParent = dadGenome.GetNeuronIndex(dadGenome.linkGenes[idx].ToNeuronId());
        if (idxInChild == -1)
        {
          childGenome.AddHiddenNeuron(dadGenome.neuronGenes[idxInParent]);
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

    for (int i = 0; i < genome1.NumLink(); ++i)
    {
      int innovId = genome1.linkGenes[i].InnovationId();
      bool linkContainedInGenome2 = genome2.ContainLink(innovId);
      if (!linkContainedInGenome2)
      {
        ++numDisjoint;
      } 
    }

    for (int i = 0; i < genome2.NumLink(); ++i)
    {
      int innovId = genome2.linkGenes[i].InnovationId();
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

    double numDisjoint = 0;

    for (int i = 0; i < genome1.NumLink(); ++i)
    {
      int linkEnabledInGenome1 = (int) genome1.linkGenes[i].Enabled();
      int innovId = genome1.linkGenes[i].InnovationId();
      int idx = genome2.GetLinkIndex(innovId);
      bool linkContainedInGenome2 = (idx != -1);

      if (linkContainedInGenome2)
      {
        int linkEnabledInGenome2 = (int) genome2.linkGenes[idx].Enabled();
        deltaW += std::abs(genome1.linkGenes[i].Weight() * linkEnabledInGenome1 - 
                           genome2.linkGenes[idx].Weight() * linkEnabledInGenome2);
        // ++coincident;
      }
      else
      {
        numDisjoint++;
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
    double delta = coeffDisjoint * deltaD + coeffWeightDiff * deltaW;

    if (delta < compatThreshold)
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
    for (int i = 0; i < population.species.size(); ++i)
    {
      if (population.species[i].genomes.size() > 0)
      {
        if (IsSameSpecies(population.species[i].genomes[0], genome))
        {  // each first genome in species is the representative genome.
          population.species[i].AddGenome(genome);
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
   * If a species has been evolved for over than staleAgeThreshold generations,
   * and its best genome's fitness doesn't change, then it is a stale species.
   * It will be removed from population.
   */
  void RemoveStaleSpecies(Population& population)
  {
    for (std::vector<Species>::iterator it = population.species.begin();
         it != population.species.end();  /*it++*/)
    {
      if(it->StaleAge() > staleAgeThreshold)
      {
        it = population.species.erase(it);
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
    for (int i = 0; i < population.species.size(); ++i)
    {
      for (int j = 0; j < population.species[i].genomes.size(); ++j)
      {
        genomes.push_back(population.species[i].genomes[j]);
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
    for (int i = 0; i < genomes.size(); ++i)
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

    for (int i = 0; i < population.species.size(); ++i)
    {
      double averageRank = 0;
      int speciesSize = population.species[i].genomes.size();
      for (int j = 0; j < speciesSize; ++j)
      {
        averageRank += genomes.size() - GetGenomeIndex(genomes, population.species[i].genomes[j].Id());
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

    for (int i = 0; i < population.species.size(); ++i)
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
    for (int i = 0; i < population.species.size(); ++i)
    {
      if (population.species[i].genomes.size() == 0)
      {
        population.species.erase(population.species.begin() + i);
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
    for (int i = 0; i < population.species.size(); ++i)
    {
      population.species[i].SortGenomes();
      int numRemove = std::floor(population.species[i].genomes.size() * percentageToRemove);
      while (numRemove > 0)
      {
        population.species[i].genomes.pop_back();
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
    for (int i = 0; i < population.species.size(); ++i)
    {
      population.species[i].SortGenomes();
      int speciesSize = population.species[i].genomes.size();
      if (speciesSize > 0)
      {
        Genome bestGenome = population.species[i].genomes[0];
        population.species[i].genomes.clear();
        population.species[i].genomes.push_back(bestGenome);
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
    MutateWeight(genome, mutateWeightProb, perturbWeightProb, mutateWeightSize);

    // Mutate add forward link.
    double p = mutateAddForwardLinkProb;
    while (p > 0)
    {  // so p can be bigger than 1 and mutate can happen multiple times.
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, FORWARD_LINK, mutateAddForwardLinkProb);
      }
      --p;
    }

    // Mutate add backward link.
    p = mutateAddBackwardLinkProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, BACKWARD_LINK, mutateAddBackwardLinkProb);
      }
      --p;
    }

    // Mutate add recurrent link.
    p = mutateAddRecurrentLinkProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, RECURRENT_LINK, mutateAddRecurrentLinkProb);
      }
      --p;
    }

    // Mutate add bias link.
    p = mutateAddBiasLinkProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddLink(genome, BIAS_LINK, mutateAddBiasLinkProb);
      }
      --p;
    }

    // Mutate add neuron.
    p = mutateAddNeuronProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateAddNeuron(genome, mutateAddNeuronProb, randomTypeNewNeuron);
      }
      --p;
    }

    // Mutate enabled node to disabled.
    p = mutateEnabledProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateEnableDisable(genome, true, mutateEnabledProb);
      }
      --p;
    }

    // Mutate disabled node to enabled.
    p = mutateDisabledProb;
    while (p > 0)
    {
      if (mlpack::math::Random() < p)
      {
        MutateEnableDisable(genome, false, mutateDisabledProb);
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
    int speciesSize = species.genomes.size();

    if (speciesSize == 0)
      return false;

    if (p < crossoverProb)
    {
      int idx1 = mlpack::math::RandInt(0, speciesSize);
      int idx2 = mlpack::math::RandInt(0, speciesSize);
      if (idx1 != idx2)
      {
        Crossover(species.genomes[idx1], species.genomes[idx2], childGenome);
      } else
      {
        return false;
      }
    } else
    {
      int idx = mlpack::math::RandInt(0, speciesSize);
      childGenome = species.genomes[idx];
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
    population = Population(seedGenome, populationSize);
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
    Genome lastBestGenome = population.BestGenome();
    childGenomes.push_back(lastBestGenome);

    // Remove weak genomes in each species.
    CullSpecies(population, cullSpeciesPercentage);

    // Remove stale species, weak species.
    if (population.species.size() > numSpeciesThreshold)
    {
      RemoveStaleSpecies(population);
      RemoveWeakSpecies(population);
    }

    // Breed children in each species. 
    std::vector<double> speciesAverageRank;
    CalcSpeciesAverageRank(population, speciesAverageRank);
    double totalAverageRank = std::accumulate(speciesAverageRank.begin(), speciesAverageRank.end(), 0);

    for (int i = 0; i < population.species.size(); ++i)
    {
      int numBreed = std::floor(speciesAverageRank[i] * populationSize / totalAverageRank) - 1;
      int numBreedSuccess = 0;

      while (numBreedSuccess < numBreed)
      {
        Genome genome;
        bool hasBaby = BreedChild(population.species[i], genome, crossoverRate);
        if (hasBaby)
        {
          childGenomes.push_back(genome);
          ++numBreedSuccess;
        }
      }
    }

    // Keep the best in each species.
    CullSpeciesToOne(population);

    // Random choose species and breed child until reach population size.
    int currentNumGenome = childGenomes.size() + population.PopulationSize();
    while (currentNumGenome < populationSize)
    {
      int speciesIndex = mlpack::math::RandInt(0, population.species.size());
      Genome genome;
      bool hasBaby = BreedChild(population.species[speciesIndex], genome, crossoverRate);
      if (hasBaby)
      {
        childGenomes.push_back(genome);
        ++currentNumGenome;
      }
    }

    // Speciate genomes into new species.
    for (int i = 0; i < childGenomes.size(); ++i)
    {
      AddGenomeToSpecies(population, childGenomes[i]);
    }

    //DEBUGGING!!!!!!!!!
    printf("Species sizes are: ");
    for (int s=0; s<population.species.size(); ++s)
    {
      std::cout<< population.species[s].genomes.size() << "  ";
    }
    printf("\n");
    //DEBUGGING!!!!!!!!!

    // Reassign genome IDs.
    population.ReassignGenomeId();
  }

  /**
   * Evaluate genomes in population.
   * Set genomes' fitness, species' and population's best fitness and genome.
   */
  template<class TaskType>
  void Evaluate(TaskType& task)
  {
    for (int i = 0; i < population.species.size(); ++i)
    {
      for (int j = 0; j < population.species[i].genomes.size(); ++j)
      {
        population.species[i].genomes[j].Flush();
      }

      // #pragma omp parallel for \
      //   shared(population) \
      //   schedule(dynamic)



      const size_t foo = population.species[i].genomes.size();
      // Genome bar = population.species[i].genomes[0];
      //#pragma omp parallel for

      // #pragma omp parallel for
      #pragma omp parallel for schedule(dynamic)
      for (int j = 0; j < foo; ++j)
      {
        // population.species[i].genomes[j].Flush();

        TaskType*  taskFoo = new TaskType();
        double fitness = taskFoo->EvalFitness(population.species[i].genomes[j]);
        // double fitness = taskFoo.EvalFitness(bar);
        population.species[i].genomes[j].Fitness(fitness);

        delete taskFoo;
      }




      double oldSpeciesBestFitness = population.species[i].BestFitness();
      population.species[i].SetBestFitnessAndGenome();
      double newSpeciesBestFitness = population.species[i].BestFitness();
      if (newSpeciesBestFitness < oldSpeciesBestFitness)
      {
        population.species[i].StaleAge(0);
      } else
      {
        int staleAge = population.species[i].StaleAge();
        population.species[i].StaleAge(staleAge + 1);
      }
    }
    population.SetBestFitnessAndGenome();
  }


  template<class TaskType>
  void Train(TaskType& task)
  {
    int generation = 0;
    // Repeat
    while (generation < maxGeneration || maxGeneration == 0)
    {
      // Evaluate all genomes in population.
      Evaluate(task);

      // Output some information.
      printf("Generation: %zu\tBest fitness: %f\n", generation, population.BestFitness());


      std::string modelSavePath = "ne_model_generation_" +
          std::to_string(generation) + ".xml";
      data::Save(modelSavePath, "ne_model", *this);

      // std::string modelRenderPath = "ne_model_generation_" +
      //     std::to_string(generation);

      // size_t trails = task.Trails();
      // bool render = task.Render();
      // std::string directory = task.Directory();

      // task.Trails() = 1;
      // task.Render() = true;
      // task.Directory() = modelRenderPath;
      // task.EvalFitness(population.BestGenome());

      // task.Trails() = trails;
      // task.Render() = render;
      // task.Directory() = directory;

      // exit(0);
      //Log::Info << "Generation: " << generation << " best fitness: " <<  population.BestFitness() << std::endl;
      if (task.Success())
      {
        printf("Task succeed in %zu iterations.\n", generation);
        break;
      }

      // Reproduce next generation.
      Reproduce();
      ++generation;
    }
  }

  void Classify()
  {

  }

  // /**
  //  * Evolve population of genomes to get a task's solution genome.
  //  *
  //  * This function is the whole progress of NEAT algorithm.
  //  */
  // bool Evolve()
  // {
  //   // Generate initial species at random.
  //   int generation = 0;
  //   InitPopulation();

  //   // Speciate genomes into species.
  //   std::vector<Genome> genomes;
  //   AggregateGenomes(population, genomes);
  //   population.species.clear();
  //   for (int i = 0; i < genomes.size(); ++i)
  //   {
  //     AddGenomeToSpecies(population, genomes[i]);
  //   }

  //   // Repeat
  //   while (generation < maxGeneration)
  //   {
  //     // Evaluate all genomes in population.
  //     Evaluate();

  //     // Output some information.
  //     printf("Generation: %zu\tBest fitness: %f\n", generation, population.BestFitness());


  //     std::string modelSavePath = "ne_model_generation_" +
  //         std::to_string(generation) + ".xml";
  //     data::Save(modelSavePath, "ne_model", *this);

  //     exit(0);
  //     //Log::Info << "Generation: " << generation << " best fitness: " <<  population.BestFitness() << std::endl;
  //     if (task.Success())
  //     {
  //       printf("Task succeed in %zu iterations.\n", generation);
  //       return true;
  //     }

  //     // Reproduce next generation.
  //     Reproduce();
  //     ++generation;
  //   }

  //   return false;
  // }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(population, "population");
    ar & data::CreateNVP(populationSize, "populationSize");
    ar & data::CreateNVP(linkInnovations, "linkInnovations");
    ar & data::CreateNVP(neuronInnovations, "neuronInnovations");
    ar & data::CreateNVP(nextNeuronId, "nextNeuronId");
    ar & data::CreateNVP(nextLinkInnovId, "nextLinkInnovId");
    ar & data::CreateNVP(maxGeneration, "maxGeneration");
    ar & data::CreateNVP(coeffDisjoint, "coeffDisjoint");
    ar & data::CreateNVP(coeffWeightDiff, "coeffWeightDiff");
    ar & data::CreateNVP(compatThreshold, "compatThreshold");
    ar & data::CreateNVP(staleAgeThreshold, "staleAgeThreshold");
    ar & data::CreateNVP(crossoverRate, "crossoverRate");
    ar & data::CreateNVP(cullSpeciesPercentage, "cullSpeciesPercentage");
    ar & data::CreateNVP(mutateWeightProb, "mutateWeightProb");
    ar & data::CreateNVP(perturbWeightProb, "perturbWeightProb");
    ar & data::CreateNVP(mutateWeightSize, "mutateWeightSize");
    ar & data::CreateNVP(mutateAddForwardLinkProb, "mutateAddForwardLinkProb");
    ar & data::CreateNVP(mutateAddBackwardLinkProb, "mutateAddBackwardLinkProb");
    ar & data::CreateNVP(mutateAddRecurrentLinkProb, "mutateAddRecurrentLinkProb");
    ar & data::CreateNVP(mutateAddBiasLinkProb, "mutateAddBiasLinkProb");
    ar & data::CreateNVP(mutateAddNeuronProb, "mutateAddNeuronProb");
    ar & data::CreateNVP(mutateEnabledProb, "mutateEnabledProb");
    ar & data::CreateNVP(mutateDisabledProb, "mutateDisabledProb");
    ar & data::CreateNVP(numSpeciesThreshold, "numSpeciesThreshold");
    ar & data::CreateNVP(randomTypeNewNeuron, "randomTypeNewNeuron");
  }

 private:


};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_NEAT_HPP
