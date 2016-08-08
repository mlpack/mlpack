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
  Genome() {
    aId = -1;
    aNumInput = 0;
    aNumOutput = 0;
    aFitness = DBL_MAX;
  }
  
  // Parametric constructor.
  Genome(int id,
  	     const std::vector<NeuronGene>& neuronGenes,
         const std::vector<LinkGene>& linkGenes,
         int numInput,
         int numOutput,
         double fitness):
    aId(id),
    aNeuronGenes(neuronGenes),
    aLinkGenes(linkGenes),
    aNumInput(numInput),
    aNumOutput(numOutput),
    aFitness(fitness)
  {}

  // Copy constructor.
  Genome(const Genome& genome) {
    aId = genome.aId;
    aNeuronGenes = genome.aNeuronGenes;
    aLinkGenes = genome.aLinkGenes;
    aNumInput = genome.aNumInput;
    aNumOutput = genome.aNumOutput;
    aFitness = genome.aFitness;
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
      aFitness = genome.aFitness;
    }

    return *this;
  }

  // Get genome id.
  int Id() const { return aId; }

  // Set genome id.
  void Id(int id) { aId = id; }

  // Get input length.
  int NumInput() { return aNumInput; }

  // Set input length.
  void NumInput(int numInput) { aNumInput = numInput; }

  // Get output length.
  int NumOutput() { return aNumOutput; }

  // Set output length.
  void NumOutput(int numOutput) { aNumOutput = numOutput; }

  // Set fitness.
  void Fitness(double fitness) { aFitness = fitness; }

  // Get fitness.
  double Fitness() const { return aFitness; }

  // Get neuron number.
  int NumNeuron() const {
    return aNeuronGenes.size();
  }
  
  // Get link number.
  int NumLink() const {
    return aLinkGenes.size();
  }

  // Get input length.
  int GetNumInput() {
    int numInput = 0;
    for (int i=0; i<aNeuronGenes.size(); ++i) {
      if (aNeuronGenes[i].Type() == INPUT || aNeuronGenes[i].Type() == BIAS)
        ++numInput;
    }
    return numInput;
  }

  // Get output length.
  int GetNumOutput() {
    int numOutput = 0;
    for (int i=0; i<aNeuronGenes.size(); ++i) {  
      if (aNeuronGenes[i].Type() == OUTPUT)
        ++numOutput;
    }
    return numOutput;
  }

  // Whether specified neuron id exist in this genome.
  bool HasNeuronId(int id) const {
    for (int i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Id() == id) {
        return true;
      }
    }
    
    return false;
  }

  // Get neuron by id.
  void GetNeuronById(int id, NeuronGene& neuronGene) {
    assert(HasNeuronId(id));

    for (int i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Id() == id) {
        neuronGene = aNeuronGenes[i];
        return;
      }
    }
  }

  // Get neuron index by id.
  int GetNeuronIndex(int id) const {
    for(int i=0; i < NumNeuron(); ++i) {
        if (aNeuronGenes[i].Id() == id) {
            return i;
        }
    }

    return -1;  // Id start from 0.
  }

  // Get link index by innovation id.
  int GetLinkIndex(int innovId) const {
    for(int i=0; i < NumLink(); ++i) {
        if (aLinkGenes[i].InnovationId() == innovId) {
            return i;
        }
    }

    return -1;  // Id start from 0.
  }

  // Whether link exist and enabled.
  bool ContainEnabledLink(int innovId) const {
    for(int i=0; i < NumLink(); ++i) {
        if (aLinkGenes[i].InnovationId() == innovId &&
            aLinkGenes[i].Enabled()) {
            return true;
        }
    }
    return false;
  }

  // Whether link exist.
  bool ContainLink(int innovId) const {
    for(int i=0; i < NumLink(); ++i) {
        if (aLinkGenes[i].InnovationId() == innovId) {
            return true;
        }
    }
    return false;
  }

  // Set neurons' input and output to zero.
  void Flush() {
    for (int i=0; i<aNeuronGenes.size(); ++i) {
      aNeuronGenes[i].Activation(0);
      aNeuronGenes[i].Input(0);
    }
  }

  // Sort link genes by toNeuron's depth.
  void SortLinkGenes() {
    struct DepthAndLink
    {
      double depth;
      LinkGene link;
 
      DepthAndLink(double d, LinkGene& l) : depth(d), link(l) {}

      bool operator < (const DepthAndLink& dL) const
      {
        return (depth < dL.depth);
      }
    };

    std::vector<double> toNeuronDepths;
    for (int i=0; i<aLinkGenes.size(); ++i) {
      NeuronGene toNeuron;
      GetNeuronById(aLinkGenes[i].ToNeuronId(), toNeuron);
      toNeuronDepths.push_back(toNeuron.Depth());
    }

    std::vector<DepthAndLink> depthAndLinks;
    int linkGenesSize = aLinkGenes.size();
    for (int i=0; i<linkGenesSize; ++i) {
      depthAndLinks.push_back(DepthAndLink(toNeuronDepths[i], aLinkGenes[i]));
    }

    std::sort(depthAndLinks.begin(), depthAndLinks.end());

    for (int i=0; i<linkGenesSize; ++i) {
      aLinkGenes[i] = depthAndLinks[i].link;
    }
  }

  // Activate genome. The last dimension of input is always 1 for bias. 0 means no bias.
  void Activate(std::vector<double>& input) {
    assert(input.size() == aNumInput);

    SortLinkGenes();
    
    // Set all neurons' input to be 0.
    for (int i=0; i<NumNeuron(); ++i) {
      aNeuronGenes[i].Input(0);
    }
    
    // Set input neurons.
    for (int i=0; i<aNumInput; ++i) {
      aNeuronGenes[i].Input(input[i]);  // assume INPUT, BIAS, OUTPUT, HIDDEN sequence
      aNeuronGenes[i].Activation(input[i]);
    }

    // Activate hidden and output neurons.
    for (int i = 0; i < NumLink(); ++i) {
      if (aLinkGenes[i].Enabled()) {
        int toNeuronIdx = GetNeuronIndex(aLinkGenes[i].ToNeuronId());
        int fromNeuronIdx = GetNeuronIndex(aLinkGenes[i].FromNeuronId());
        double input = aNeuronGenes[toNeuronIdx].Input() + 
                       aNeuronGenes[fromNeuronIdx].Activation() * aLinkGenes[i].Weight();
        aNeuronGenes[toNeuronIdx].Input(input);
        
        if (i == NumLink() - 1) {
          aNeuronGenes[toNeuronIdx].CalcActivation();
        } else if (GetNeuronIndex(aLinkGenes[i + 1].ToNeuronId()) != toNeuronIdx) {
          aNeuronGenes[toNeuronIdx].CalcActivation();
        }
      }
    }
  }

  // Get output vector.
  void Output(std::vector<double>& output) {
    output.clear();
    for (int i=0; i<aNumOutput; ++i) {
      output.push_back(aNeuronGenes[aNumInput + i].Activation());
    }
  }

  // Set random link weights between [lo, hi].
  void RandomizeWeights(const double lo, const double hi) {
    for (int i=0; i<aLinkGenes.size(); ++i) {
      double weight = mlpack::math::Random(lo, hi);
      aLinkGenes[i].Weight(weight); 
    }
  }

  // Add link.
  void AddLink(LinkGene& linkGene) {
    aLinkGenes.push_back(linkGene);
  }

  // Add neuron.
  void AddHiddenNeuron(NeuronGene& neuronGene) {
    if (neuronGene.Type() == HIDDEN) {
      aNeuronGenes.push_back(neuronGene);
    }
  }

  // Show genome structure.
  void Show() {
    
  }

 private:
  // Genome id.
  int aId;

  // Input length (include bias). 
  int aNumInput;

  // Output length.
  int aNumOutput;

  // Genome fitness.
  double aFitness;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENOME_HPP
