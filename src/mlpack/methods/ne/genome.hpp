/**
 * @file genome.hpp
 * @author Bang Liu
 *
 * Definition of the Genome class.
 */
#ifndef MLPACK_METHODS_NE_GENOME_HPP
#define MLPACK_METHODS_NE_GENOME_HPP

#include <cstddef>
#include <cassert>
#include <map>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "utils.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a genome.
 # A genome is consist of a group of genes.
 */
class Genome {
 public:
  // Neurons.
  std::vector<NeuronGene> aNeuronGenes;

  // Links.
  std::vector<LinkGene> aLinkGenes;

  // Default constructor.
  Genome() {}
  
  // Parametric constructor.
  Genome(ssize_t id,
  	     const std::vector<NeuronGene>& neuronGenes,
         const std::vector<LinkGene>& linkGenes,
         ssize_t numInput,
         ssize_t numOutput,
         ssize_t depth,
         double fitness,
         double adjustedFitness):
    aId(id),
    aNeuronGenes(neuronGenes),
    aLinkGenes(linkGenes),
    aNumInput(numInput),
    aNumOutput(numOutput),
    aDepth(depth),
    aFitness(fitness),
    aAdjustedFitness(adjustedFitness)
  {}

  // Copy constructor.
  Genome(const Genome& genome) {
    aId = genome.aId;
    aNeuronGenes = genome.aNeuronGenes;
    aLinkGenes = genome.aLinkGenes;
    aNumInput = genome.aNumInput;
    aNumOutput = genome.aNumOutput;
    aDepth = genome.aDepth;
    aFitness = genome.aFitness;
    aAdjustedFitness = genome.aAdjustedFitness;
  }

  // Destructor.
  ~Genome() {}

  // Operator =.
  Genome& operator =(const Genome& genome) {
    if (this != &genome) {
      aId = genome.aId;
      aNeuronGenes = genome.aNeuronGenes;
      aLinkGenes = genome.aLinkGenes;
      aNumInput = genome.aNumInput;
      aNumOutput = genome.aNumOutput;
      aDepth = genome.aDepth;
      aFitness = genome.aFitness;
      aAdjustedFitness = genome.aAdjustedFitness;
    }

    return *this;
  }

  // Get genome id.
  ssize_t Id() const { return aId; }

  // Set genome id.
  void Id(ssize_t id) { aId = id; }

  // Get input length.
  ssize_t NumInput() const { return aNumInput; }

  // Set input length.
  void NumInput(ssize_t numInput) { aNumInput = numInput; }

  // Get output length.
  ssize_t NumOutput() const { return aNumOutput; }

  // Set output length.
  void NumOutput(ssize_t numOutput) { aNumOutput = numOutput; }

  // Get depth.
  ssize_t Depth() const { return aDepth; }

  // Set depth.
  void Depth(ssize_t depth) { aDepth = depth; }

  // Set fitness.
  void Fitness(double fitness) { aFitness = fitness; }

  // Get fitness.
  double Fitness() const { return aFitness; }

  // Set adjusted fitness.
  void AdjustedFitness(double adjustedFitness) { aAdjustedFitness = adjustedFitness; }

  // Get adjusted fitness.
  double AdjustedFitness() const { return aAdjustedFitness; }

  // Get neuron number.
  ssize_t NumNeuron() const {
    return aNeuronGenes.size();
  }
  
  // Get link number.
  ssize_t NumLink() const {
    return aLinkGenes.size();
  }

  // Whether specified neuron id exist in this genome.
  bool HasNeuronId(ssize_t id) const {
    assert(id > 0);
    assert(NumNeuron() > 0);

    for (ssize_t i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Id() == id) {
        return true;
      }
    }
    
    return false;
  }

  // Get neuron by id.
  NeuronGene GetNeuronById(ssize_t id) {
    assert(HasNeuronId(id));

    for (ssize_t i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Id() == id) {
        return aNeuronGenes[i];
      }
    }
  }

  // Get neuron index by id.
  ssize_t GetNeuronIndex(ssize_t id) const {
    for(ssize_t i=0; i < NumNeuron(); ++i) {
        if (aNeuronGenes[i].Id() == id) {
            return i;
        }
    }

    return -1;  // Id start from 0.
  }

  // Calculate Neuron depth.
  ssize_t NeuronDepth(ssize_t id) {
    // TODO: if contains loop in network.

    // Find all links that output to this neuron id.
    std::vector<int> inputLinksIndex;
    for (ssize_t i=0; i<NumLink(); ++i) {
      if (aLinkGenes[i].ToNeuronId() == id) {
        inputLinksIndex.push_back(i);
      }
    }

    // INPUT or BIAS or isolated neurons.
    if (inputLinksIndex.size() == 0) {
      return 0;
    }

    // Recursively get neuron depth.
    std::vector<ssize_t> depths(inputLinksIndex.size());
    for (ssize_t i=0; i<inputLinksIndex.size(); ++i) {
      depths[i] = NeuronDepth(aLinkGenes[inputLinksIndex[i]].FromNeuronId());
    }
    ssize_t maxInputDepth = *(std::max_element(std::begin(depths),
                                 std::end(depths)));
    return (maxInputDepth + 1);
  }

  // Calculate Genome depth.
  // It is the max depth of all output neuron genes.
  ssize_t GenomeDepth() {
    ssize_t numNeuron = NumNeuron();

    // If empty genome.
    if (numNeuron == 0) {
      aDepth = 0;
      return aDepth;
    }

    // If no hidden neuron, depth is 1.
    if (aNumInput + aNumOutput == numNeuron) {
      aDepth = 1;
      return aDepth;
    }
    
    // Find all OUTPUT neuron id.
    std::vector<ssize_t> outputNeuronsId;
    for (ssize_t i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Type() == OUTPUT) {
        outputNeuronsId.push_back(aNeuronGenes[i].Id());
      }
    }

    // Get max depth of all output neurons.
    ssize_t genomeDepth = 0;
    for (ssize_t i=0; i<outputNeuronsId.size(); ++i) {
      ssize_t outputNeuronDepth = NeuronDepth(outputNeuronsId[i]);
      if (outputNeuronDepth > genomeDepth) {
        genomeDepth = outputNeuronDepth;
      }
    }
    aDepth = genomeDepth;

    return aDepth;
  }

  // Set neurons' input and output to zero.
  void Flush() {
    for (ssize_t i=0; i<aNeuronGenes.size(); ++i) {
      aNeuronGenes[i].aActivation = 0;
      aNeuronGenes[i].aInput = 0;
    }
  }

  // Activate genome. The last dimension of input is always 1 for bias. 0 means no bias.
  void Activate(std::vector<double>& input) {
    assert(input.size() == aNumInput);
    //Flush();

    // Set inputs.
    for (ssize_t i=0; i<aNumInput; ++i) {
      aNeuronGenes[i].aActivation = input[i];  // assume INPUT, BIAS, OUTPUT, HIDDEN sequence
    }

    // Construct neuron id: index dictionary.
    std::map<ssize_t, ssize_t> neuronIdToIndex;
    for (ssize_t i=0; i<NumNeuron(); ++i) {
      neuronIdToIndex.insert(std::pair<ssize_t, ssize_t>(aNeuronGenes[i].Id(), i));
    }

    // Activate layer by layer.
    for (ssize_t i=0; i<aDepth; ++i) {
      // Loop links to calculate neurons' input sum.
      for (ssize_t j=0; j<aLinkGenes.size(); ++j) {
        aNeuronGenes[neuronIdToIndex.at(aLinkGenes[j].ToNeuronId())].aInput +=
          aLinkGenes[j].Weight() *
          aNeuronGenes[neuronIdToIndex.at(aLinkGenes[j].FromNeuronId())].aActivation *
          ((int) aLinkGenes[j].Enabled());
      }

      // Loop neurons to calculate neurons' activation.
      for (ssize_t j=aNumInput; j<aNeuronGenes.size(); ++j) {
        double x = aNeuronGenes[j].aInput;  // TODO: consider bias. Difference?
        aNeuronGenes[j].aInput = 0;

        double y = 0;
        switch (aNeuronGenes[j].Type()) { // TODO: more cases.
          case SIGMOID:                   
            y = ann::LogisticFunction::fn(x);
            break;
          case TANH:
            y = ann::TanhFunction::fn(x);
            break;
          case RELU:
            y = ann::RectifierFunction::fn(x);
            break;
          case LINEAR:
            y = x;
          default:
            y = ann::LogisticFunction::fn(x);
            break;
        }
        aNeuronGenes[j].aActivation = y;
      }
    }
  }

  // Get output vector.
  std::vector<double> Output() {
    std::vector<double> output;
    for (ssize_t i=0; i<aNumOutput; ++i) {  // TODO: it relies on the sequence. Good?
      output.push_back(aNeuronGenes[aNumInput + i].aActivation);
    }
    return output;
  }

  // Set random link weights between [lo, hi].
  void RandomizeWeights(const double lo, const double hi) {
    for (ssize_t i=0; i<aLinkGenes.size(); ++i) {
      double weight = mlpack::math::Random(lo, hi);
      aLinkGenes[i].Weight(weight); 
    }
  }

  // Soft mutation: add a random value chosen from
  // initialization prob distribution with probability p.
  // TODO: here we use uniform distribution.
  // TODO: make genome an parameter and put it into CNE??
  // Can we use exponential distribution?
  void MutateWeightsBiased(double mutateProb, double mutateSize) {
    for (ssize_t i=0; i<aLinkGenes.size(); ++i) {
      double p = mlpack::math::Random();  // rand 0~1
      if (p < mutateProb) {
        double deltaW = mlpack::math::RandNormal(0, mutateSize);  // TODO: make it tunable: gaussian, param, exp-dist...
        double oldW = aLinkGenes[i].Weight();
        aLinkGenes[i].Weight(oldW + deltaW);
      }
    }
  }

  // Hard mutation: replace with a random value chosen from
  // initialization prob distribution with probability p.
  // TODO: here we use uniform distribution.
  // TODO: make genome an parameter and put it into CNE??
  // Can we use exponential distribution?
  void MutateWeightsUnbiased(double mutateProb, double mutateSize) {
    for (ssize_t i=0; i<aLinkGenes.size(); ++i) {
      double p = mlpack::math::Random();
      if (p < mutateProb) {
        double weight = mlpack::math::RandNormal(0, mutateSize); // TODO: make it tunable: gaussian, param, exp-dist...
        aLinkGenes[i].Weight(weight);
      }
    }
  }

  // Randomly select weights from one parent genome.
  static void CrossoverWeights(Genome& momGenome, 
                               Genome& dadGenome, 
                               Genome& child1Genome, 
                               Genome& child2Genome) {
    child1Genome = momGenome;
    child2Genome = dadGenome;
    for (ssize_t i=0; i<momGenome.aLinkGenes.size(); ++i) { // assume genome are the same structure.
      double t = mlpack::math::RandNormal();
      if (t>0) {  // prob = 0.5
        child1Genome.aLinkGenes[i].Weight(momGenome.aLinkGenes[i].Weight());
        child2Genome.aLinkGenes[i].Weight(dadGenome.aLinkGenes[i].Weight());
      } else {
        child1Genome.aLinkGenes[i].Weight(dadGenome.aLinkGenes[i].Weight());
        child2Genome.aLinkGenes[i].Weight(momGenome.aLinkGenes[i].Weight());
      }
    }

    // NOTICE: child genomes need to set genome id based on its population's max id.
    // TODO: check this works well.
  }

  // TODO: More potential operators to search genome.


 private:
  // Genome id.
  ssize_t aId;

  // Input length (include bias). 
  ssize_t aNumInput;

  // Output length.
  ssize_t aNumOutput;

  // Network maximum depth.
  ssize_t aDepth;

  // Genome fitness.
  double aFitness;

  // Genome adjusted fitness.
  double aAdjustedFitness;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENOME_HPP
