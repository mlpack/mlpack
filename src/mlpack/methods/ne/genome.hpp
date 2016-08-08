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

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "utils.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a genome.
 * A genome is consist of a group of genes.
 */
class Genome 
{
 public:
  //! Neurons.
  std::vector<NeuronGene> aNeuronGenes;

  //! Links.
  std::vector<LinkGene> aLinkGenes;

  /**
   * Default constructor.
   */
  Genome(): 
    aId(-1),
    aNumInput(0),
    aNumOutput(0),
    aFitness(DBL_MAX)
  {}
  
  /**
   * Construct the Genome object with the given parameters.
   *
   * @param id Genome identifier.
   * @param neuronGenes List of genes to construct the genome.
   * @param linkGenes List of links to construct the genome.
   * @param numInput Number of input genes.
   * @param numOutput Number of output genes.
   * @param fitness Initial fitness.
   */
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

  /**
   * Copy constructor.
   *
   * @param genome The copied genome.
   */
  Genome(const Genome& genome)
  {
    aId = genome.aId;
    aNeuronGenes = genome.aNeuronGenes;
    aLinkGenes = genome.aLinkGenes;
    aNumInput = genome.aNumInput;
    aNumOutput = genome.aNumOutput;
    aFitness = genome.aFitness;
  }

  /**
   * Operator =.
   *
   * @param genome The genome to be compared with.
   */
  Genome& operator =(const Genome& genome)
  {
    if (this != &genome)
    {
      aId = genome.aId;
      aNeuronGenes = genome.aNeuronGenes;
      aLinkGenes = genome.aLinkGenes;
      aNumInput = genome.aNumInput;
      aNumOutput = genome.aNumOutput;
      aFitness = genome.aFitness;
    }

    return *this;
  }

  /**
   * Get genome id.
   */
  int Id() const { return aId; }

  /**
   * Set genome id.
   */
  void Id(int id) { aId = id; }

  /**
   * Get input length.
   */
  int NumInput() { return aNumInput; }

  /**
   * Set input length.
   */
  void NumInput(int numInput) { aNumInput = numInput; }

  /**
   * Get output length.
   */
  int NumOutput() { return aNumOutput; }

  /**
   * Set output length.
   */
  void NumOutput(int numOutput) { aNumOutput = numOutput; }

  /**
   * Set fitness.
   */
  void Fitness(double fitness) { aFitness = fitness; }

  /**
   * Get fitness.
   */
  double Fitness() const { return aFitness; }

  /**
   * Get neuron number.
   */
  int NumNeuron() const { return aNeuronGenes.size(); }
  
  /**
   * Get link number.
   */
  int NumLink() const { return aLinkGenes.size(); }

  /**
   * Get input length.
   */
  int GetNumInput()
  {
    int numInput = 0;
    for (int i=0; i<aNeuronGenes.size(); ++i)
    {
      if (aNeuronGenes[i].Type() == INPUT || aNeuronGenes[i].Type() == BIAS)
        ++numInput;
    }
    return numInput;
  }

  /**
   * Get output length.
   */
  int GetNumOutput()
  {
    int numOutput = 0;
    for (int i=0; i<aNeuronGenes.size(); ++i)
    {  
      if (aNeuronGenes[i].Type() == OUTPUT)
        ++numOutput;
    }
    return numOutput;
  }

  /**
   * Whether specified neuron id exist in this genome.
   *
   * @param id Check whether id exist in this genome.
   */
  bool HasNeuronId(int id) const
  {
    for (int i=0; i<NumNeuron(); ++i)
    {
      if (aNeuronGenes[i].Id() == id)
      {
        return true;
      }
    }
    
    return false;
  }

  /**
   * Get neuron by id.
   *
   * @param id The id of the retrieved neuron.
   * @param neuronGene Return parameter. The retrieved neuron.
   */
  void GetNeuronById(int id, NeuronGene& neuronGene)
  {
    assert(HasNeuronId(id));

    for (int i=0; i<NumNeuron(); ++i)
    {
      if (aNeuronGenes[i].Id() == id)
      {
        neuronGene = aNeuronGenes[i];
        return;
      }
    }
  }

  /**
   * Get neuron index by id.
   *
   * @param id Neuron's id.
   */
  int GetNeuronIndex(int id) const
  {
    for(int i=0; i < NumNeuron(); ++i)
    {
        if (aNeuronGenes[i].Id() == id)
        {
            return i;
        }
    }

    return -1;  // Id start from 0.
  }

  /**
   * Get link index by innovation id.
   *
   * @param innovId Link's innovation id.
   */
  int GetLinkIndex(int innovId) const
  {
    for(int i=0; i < NumLink(); ++i)
    {
        if (aLinkGenes[i].InnovationId() == innovId)
        {
            return i;
        }
    }

    return -1;  // Id start from 0.
  }

  /**
   * Whether a link exist and enabled.
   *
   * @param innovId Link's innovation id.
   */
  bool ContainEnabledLink(int innovId) const
  {
    for(int i=0; i < NumLink(); ++i)
    {
        if (aLinkGenes[i].InnovationId() == innovId &&
            aLinkGenes[i].Enabled())
        {
            return true;
        }
    }
    return false;
  }

  /**
   * Whether link exist.
   *
   * @param innovId Link's innovation id.
   */
  bool ContainLink(int innovId) const
  {
    for(int i=0; i < NumLink(); ++i)
    {
        if (aLinkGenes[i].InnovationId() == innovId)
        {
            return true;
        }
    }
    return false;
  }

  /**
   * Set all the neurons' input and output to be zero.
   */
  void Flush()
  {
    for (int i=0; i<aNeuronGenes.size(); ++i)
    {
      aNeuronGenes[i].Activation(0);
      aNeuronGenes[i].Input(0);
    }
  }

  /**
   * Sort link genes by toNeuron's depth.
   *
   * Links will be sort by the value of its toNeuron's depth,
   * from 0 (input) to 1 (output). This will be helpful for the
   * calculation of genome's activation.
   */
  void SortLinkGenes()
  {
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
    for (int i=0; i<aLinkGenes.size(); ++i)
    {
      NeuronGene toNeuron;
      GetNeuronById(aLinkGenes[i].ToNeuronId(), toNeuron);
      toNeuronDepths.push_back(toNeuron.Depth());
    }

    std::vector<DepthAndLink> depthAndLinks;
    int linkGenesSize = aLinkGenes.size();
    for (int i=0; i<linkGenesSize; ++i)
    {
      depthAndLinks.push_back(DepthAndLink(toNeuronDepths[i], aLinkGenes[i]));
    }

    std::sort(depthAndLinks.begin(), depthAndLinks.end());

    for (int i=0; i<linkGenesSize; ++i)
    {
      aLinkGenes[i] = depthAndLinks[i].link;
    }
  }

  /**
   * Activate genome.
   * 
   * Calculate genome's output given input.
   * The last dimension of input is always 1 for bias. If 0, then means no bias.
   *
   * @param input Input of the genome.
   */
  void Activate(std::vector<double>& input)
  {
    assert(input.size() == aNumInput);

    SortLinkGenes();
    
    // Set all neurons' input to be 0.
    for (int i=0; i<NumNeuron(); ++i)
    {
      aNeuronGenes[i].Input(0);
    }
    
    // Set input neurons.
    for (int i=0; i<aNumInput; ++i)
    {
      aNeuronGenes[i].Input(input[i]);  // assume INPUT, BIAS, OUTPUT, HIDDEN sequence
      aNeuronGenes[i].Activation(input[i]);
    }

    // Activate hidden and output neurons.
    for (int i = 0; i < NumLink(); ++i)
    {
      if (aLinkGenes[i].Enabled())
      {
        int toNeuronIdx = GetNeuronIndex(aLinkGenes[i].ToNeuronId());
        int fromNeuronIdx = GetNeuronIndex(aLinkGenes[i].FromNeuronId());
        double input = aNeuronGenes[toNeuronIdx].Input() + 
                       aNeuronGenes[fromNeuronIdx].Activation() * aLinkGenes[i].Weight();
        aNeuronGenes[toNeuronIdx].Input(input);
        
        if (i == NumLink() - 1)
        {
          aNeuronGenes[toNeuronIdx].CalcActivation();
        } else if (GetNeuronIndex(aLinkGenes[i + 1].ToNeuronId()) != toNeuronIdx)
        {
          aNeuronGenes[toNeuronIdx].CalcActivation();
        }
      }
    }
  }

  /**
   * Get output vector.
   *
   * @param output Return parameter, store the output vector.
   */
  void Output(std::vector<double>& output)
  {
    output.clear();
    for (int i=0; i<aNumOutput; ++i)
    {
      output.push_back(aNeuronGenes[aNumInput + i].Activation());
    }
  }

  /**
   * Set random link weights between [lo, hi].
   *
   * @param lo Low bound of random weight.
   * @param hi High bound of random weight.
   */
  void RandomizeWeights(const double lo, const double hi)
  {
    for (int i=0; i<aLinkGenes.size(); ++i)
    {
      double weight = mlpack::math::Random(lo, hi);
      aLinkGenes[i].Weight(weight); 
    }
  }

  /**
   * Add link to link list.
   *
   * @param linkGene The new link to add.
   */
  void AddLink(LinkGene& linkGene)
  {
    aLinkGenes.push_back(linkGene);
  }

  /**
   * Add hidden neuron to neuron list.
   *
   * @param neuronGene The new hidden neuron to add.
   */
  void AddHiddenNeuron(NeuronGene& neuronGene)
  {
    if (neuronGene.Type() == HIDDEN)
    {
      aNeuronGenes.push_back(neuronGene);
    }
  }

 private:
  //! Genome id.
  int aId;

  //! Input length (include bias). 
  int aNumInput;

  //! Output length.
  int aNumOutput;

  //! Genome fitness.
  double aFitness;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_GENOME_HPP
