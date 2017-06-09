#ifndef MLPACK_METHODS_NEURO_CMAES_GENOME_HPP
#define MLPACK_METHODS_NEURO_CMAES_GENOME_HPP

#include <cstddef>
#include <cassert>
#include <map>

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"

namespace mlpack {
namespace neuro_cmaes {

/**
 * This class defines a genome.
 # A genome which uses neurons and links class to make its structure.
 */
class Genome {
 public:
  // Neurons.
  std::vector<NeuronGene> aNeuronGenes;

  // Links.
  std::vector<LinkGene> aLinkGenes;

  // Default constructor.
  Genome()
   {
    aNumInput = 0;
    aNumOutput = 0;
   }
  
  // Parametric constructor.
  Genome(const std::vector<NeuronGene>& neuronGenes,
         const std::vector<LinkGene>& linkGenes,
         ssize_t numInput,
         ssize_t numOutput):
         
    aNeuronGenes(neuronGenes),
    aLinkGenes(linkGenes),
    aNumInput(numInput),
    aNumOutput(numOutput)
  {}

  // Copy constructor.
  Genome(const Genome& genome) 
  {
    aNeuronGenes = genome.aNeuronGenes;
    aLinkGenes = genome.aLinkGenes;
    aNumInput = genome.aNumInput;
    aNumOutput = genome.aNumOutput;
  }

  // Destructor.
  ~Genome() {}

  // Operator =.
  Genome& operator =(const Genome& genome) {
    if (this != &genome) 
    {
      aNeuronGenes = genome.aNeuronGenes;
      aLinkGenes = genome.aLinkGenes;
      aNumInput = genome.aNumInput;
      aNumOutput = genome.aNumOutput;
    }

    return *this;
  }

  // Get input length.
  ssize_t NumInput() { return aNumInput; }

  // Set input length.
  void NumInput(ssize_t numInput) { aNumInput = numInput; }

  // Get output length.
  ssize_t NumOutput() { return aNumOutput; }

  // Set output length.
  void NumOutput(ssize_t numOutput) { aNumOutput = numOutput; }

  // Get neuron number.
  ssize_t NumNeuron() const {
    return aNeuronGenes.size();
  }
  
  // Get link number.
  ssize_t NumLink() const {
    return aLinkGenes.size();
  }

  // Get input length.
  ssize_t GetNumInput() {
    ssize_t numInput = 0;
    for (ssize_t i=0; i<aNeuronGenes.size(); ++i) {
      if (aNeuronGenes[i].Type() == INPUT || aNeuronGenes[i].Type() == BIAS)
        ++numInput;
    }
    return numInput;
  }

  // Get output length.
  ssize_t GetNumOutput() {
    ssize_t numOutput = 0;
    for (ssize_t i=0; i<aNeuronGenes.size(); ++i) {  
      if (aNeuronGenes[i].Type() == OUTPUT)
        ++numOutput;
    }
    return numOutput;
  }

  // Whether specified neuron id exist in this genome.
  bool HasNeuronId(ssize_t id) const {
    for (ssize_t i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Id() == id) {
        return true;
      }
    }
    
    return false;
  }

  // Get neuron by id.
  void GetNeuronById(ssize_t id, NeuronGene& neuronGene) {
    assert(HasNeuronId(id));

    for (ssize_t i=0; i<NumNeuron(); ++i) {
      if (aNeuronGenes[i].Id() == id) {
        neuronGene = aNeuronGenes[i];
        return;
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

   void Flush() 
   {
    for (ssize_t i=0; i<aNeuronGenes.size(); ++i)
     {
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
    for (ssize_t i=0; i<aLinkGenes.size(); ++i) {
      NeuronGene toNeuron;
      GetNeuronById(aLinkGenes[i].ToNeuronId(), toNeuron);
      toNeuronDepths.push_back(toNeuron.Depth());
    }

    std::vector<DepthAndLink> depthAndLinks;
    ssize_t linkGenesSize = aLinkGenes.size();
    for (ssize_t i=0; i<linkGenesSize; ++i) {
      depthAndLinks.push_back(DepthAndLink(toNeuronDepths[i], aLinkGenes[i]));
    }

    std::sort(depthAndLinks.begin(), depthAndLinks.end());

    for (ssize_t i=0; i<linkGenesSize; ++i) {
      aLinkGenes[i] = depthAndLinks[i].link;
    }
  }


  // Activate genome. The last dimension of input is always 1 for bias. 0 means no bias.
  void Activate(std::vector<double>& input) {
    assert(input.size() == aNumInput);
    
    // Set all neurons' input to be 0.
    for (ssize_t i=0; i<NumNeuron(); ++i) {
      aNeuronGenes[i].Input(0);
    }
    
    // Set input neurons.
    for (ssize_t i=0; i<aNumInput; ++i) {
      aNeuronGenes[i].Input(input[i]);  // assume INPUT, BIAS, OUTPUT, HIDDEN sequence
      aNeuronGenes[i].Activation(input[i]);
    }

    // Activate hidden and output neurons.
    for (ssize_t i = 0; i < NumLink(); ++i) 
    {
     
        ssize_t toNeuronIdx = GetNeuronIndex(aLinkGenes[i].ToNeuronId());
        ssize_t fromNeuronIdx = GetNeuronIndex(aLinkGenes[i].FromNeuronId());
        double input = aNeuronGenes[toNeuronIdx].Input() + 
                       aNeuronGenes[fromNeuronIdx].Activation() * aLinkGenes[i].Weight();
        aNeuronGenes[toNeuronIdx].Input(input);
        
        if (i == NumLink() - 1) 
        {
          aNeuronGenes[toNeuronIdx].CalcActivation();
        }
         else if (GetNeuronIndex(aLinkGenes[i + 1].ToNeuronId()) != toNeuronIdx) 
        {
          aNeuronGenes[toNeuronIdx].CalcActivation();
        }
      
    }
  }

  // Get output vector.
  void Output(std::vector<double>& output) {
    output.clear();
    for (ssize_t i=0; i<aNumOutput; ++i) {
      output.push_back(aNeuronGenes[aNumInput + i].Activation());
    }
  }


 private:
  // Input length (include bias). 
  ssize_t aNumInput;

  // Output length.
  ssize_t aNumOutput;

};

}  // namespace neuro_cmaes
}  // namespace mlpack

#endif  // MLPACK_METHODS_NEURO_CMAES_GENOME_HPP
