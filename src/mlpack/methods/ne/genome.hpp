/**
 * @file genome.hpp
 * @author Bang Liu
 *
 * Definition of the Genome class.
 */
#ifndef MLPACK_METHODS_NE_GENOME_HPP
#define MLPACK_METHODS_NE_GENOME_HPP

#include <cassert>
#include <map>

#include <mlpack/core.hpp>

#include "gene.hpp"
#include "utils.hpp"

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
         unsigned int numInput,
         unsigned int numOutput,
         unsigned int depth,
         double fitness):
    aId(id),
    aNeuronGenes(neuronGenes),
    aLinkGenes(linkGenes),
    aNumInput(numInput),
    aNumOutput(numOutput),
    aDepth(depth),
    aFitness(fitness)
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
  }

  // Destructor.
  ~Genome() {}

  // Get genome id.
  unsigned int Id() const { return aId; }

  // Get input length.
  unsigned int NumInput() const { return aNumInput; }

  // Set input length.
  void NumInput(unsigned int numInput) { aNumInput = numInput; }

  // Get output length.
  unsigned int NumOutput() const { return aNumOutput; }

  // Set output length.
  void NumOutput(unsigned int numOutput) { aNumOutput = numOutput; }

  // Get depth.
  unsigned int Depth() const { return aDepth; }

  // Set depth.
  void Depth(unsigned int depth) { aDepth = depth; }

  // Get fitness.
  double Fitness() const { return aFitness; }

  // Set fitness.
  void Fitness(double fitness) { aFitness = fitness; }

  // Get neuron number.
  unsigned int NumNeuron() const {
    return static_cast<unsigned int>(aNeuronGenes.size());
  }
  
  // Get link number.
  unsigned int NumLink() const {
    return static_cast<unsigned int>(aLinkGenes.size());
  }

  // Whether specified neuron id exist in this genome.
  bool HasNeuronId(unsigned int id) const {
    assert(id > 0);
    assert(NumNeuron() > 0);

    for (unsigned int i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].aId == id) {
        return true;
      }
    }
    
    return false;
  }

  // Get neuron by id.
  NeuronGene GetNeuronById(unsigned int id) {
    assert(HasNeuronId(id));

    for (unsigned int i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].aId == id) {
        return aNeuronGenes[i];
      }
    }
  }

  // Get neuron index by id.
  unsigned int GetNeuronIndex(unsigned int id) const {
    for(unsigned int i=0; i < NumNeuron(); ++i) {
        if (aNeuronGenes[i].Id() == id) {
            return i;
        }
    }

    return -1;
  }

  // Calculate Neuron depth.
  unsigned int NeuronDepth(unsigned int id) {
    // TODO: if contains loop in network.

    // Find all links that output to this neuron id.
    std::vector<int> inputLinksIndex;
    for (unsigned int i=0; i<NumLink(); ++i) {
      if (aLinkGenes[i].ToNeuronId() == id) {
        inputLinksIndex.push_back(i);
      }
    }

    // INPUT or BIAS or isolated neurons.
    if (inputLinksIndex.size() == 0) {
      return 0;
    }

    // Recursively get neuron depth.
    std::vector<unsigned int> depths(inputLinksIndex.size());
    for (unsigned int i=0; i<inputLinksIndex.size(); ++i) {
      depths[i] = NeuronDepth(aLinkGenes[inputLinksIndex[i]].FromNeuronId());
    }
    unsigned int maxInputDepth = *(std::max_element(std::begin(depths),
                                 std::end(depths)));
    return (maxInputDepth + 1);
  }

  // Calculate Genome depth.
  // It is the max depth of all output neuron genes.
  unsigned int GenomeDepth() {
    unsigned int numNeuron = NumNeuron();

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
    std::vector<unsigned int> outputNeuronsId;
    for (unsigned int i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Type() == OUTPUT) {
        outputNeuronsId.push_back(aNeuronGenes[i].Id());
      }
    }

    // Get max depth of all output neurons.
    unsigned int genomeDepth = 0;
    for (unsigned int i=0; i<outputNeuronsId.size(); ++i) {
      unsigned int outputNeuronDepth = NeuronDepth(outputNeuronsId[i]);
      if (outputNeuronDepth > genomeDepth) {
        genomeDepth = outputNeuronDepth;
      }
    }
    aDepth = genomeDepth;

    return aDepth;
  }

  // Set neurons' input and output to zero.
  void Flush() {
    for (unsigned int i=0; i<aNeuronGenes.size(); ++i) {
      aNeuronGenes[i].aActivation = 0;
      aNeuronGenes[i].aInput = 0;
    }
  }

  // Activate genome. The last dimension of input is always 1 for bias. 0 means no bias.
  void Activate(std::vector<double>& input) {
    assert(input.size() == aNumInput);
    Flush();

    // Set inputs.
    for (unsigned int i=0; i<aNumInput; ++i) {
      aNeuronGenes[i].aActivation = input[i];  // assume INPUT, BIAS, OUTPUT, HIDDEN sequence
    }

    // Construct neuron id: index dictionary.
    std::map<unsigned int, unsigned int> neuronIdToIndex;
    for (unsigned int i=0; i<NumNeuron(); ++i) {
      neuronIdToIndex.insert(std::pair<unsigned int, unsigned int>(aNeuronGenes[i].Id(), i));
    }

    // Activate layer by layer.
    for (unsigned int i=0; i<aDepth; ++i) {
      // Loop links to calculate neurons' input sum.
      for (unsigned int j=0; j<aLinkGenes.size(); ++j) {
        aNeuronGenes[neuronIdToIndex.at(aLinkGenes[j].ToNeuronId())].aInput +=
          aLinkGenes[j].Weight() * aNeuronGenes[neuronIdToIndex.at(aLinkGenes[j].FromNeuronId())].aActivation;
      }

      // Loop neurons to calculate neurons' activation.
      for (unsigned in j=aNumInput; j<aNeuronGenes.size(); ++j) {
        double x = aNeuronGenes[j].aInput;  // TODO: consider bias. Difference?
        aNeuronGenes[j].aInput = 0;

        double y = 0;
        switch (aNeuronGenes[j].Type()) { // TODO: revise the implementation.
          case SIGMOID:                   // TODO: more cases.
            y = sigmoid(x);
            break;
          case RELU:
            y = relu(x);
            break;
          default:
            y = sigmoid(x);
            break;
        }
        aNeuronGenes[j].aActivation = y;
      }
    }
  }

  // Get output vector.
  std::vector<double> Output() {
    std::vector<double> output;
    for (int i=0; i<aNumOutput; ++i) {  // TODO: it relies on the sequence. Good?
      output.push_back(aNeuronGenes[aNumInput + i].aActivation);
    }
    return output;
  }

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

  // Input length (include bias). 
  unsigned int aNumInput;

  // Output length.
  unsigned int aNumOutput;

  // Network maximum depth.
  unsigned int aDepth;

  // Genome fitness.
  double aFitness;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENOME_HPP