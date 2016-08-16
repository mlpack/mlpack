/**
 * @file substrate.hpp
 * @author Bang Liu
 *
 * Definition of the Substrate class.
 */
#ifndef MLPACK_METHODS_NE_SUBSTRATE_HPP
#define MLPACK_METHODS_NE_SUBSTRATE_HPP

#include <mlpack/core.hpp>

#include "link_gene.hpp"
#include "neuron_gene.hpp"
#include "genome.hpp"

namespace mlpack {
namespace ne {

/**
 * This class defines a substrate.
 */
class Substrate
{
 public:
  //! Coordinates of nodes. Sequence is INPUT, BIAS, OUTPUT, HIDDEN
  std::vector< std::vector<double> > aCoordinates;

  //! Depths of nodes.
  std::vector<double> aDepths;

  //! Number of input nodes. Including one bias node.
  int aNumInput;

  //! Number of output nodes.
  int aNumOutput;

  //! Connection allowed or not.
  //! if aAllowedConnectionMask[i][j] == 0, then link between node i to j is allowed.
  //! Otherwise, link not allowed. i, j is neuron id.
  std::vector< std::vector<int> > aAllowedConnectionMask;

  //! The threshold of weight to create new connection.
  double aWeightThreshold;

  /**
   * Default constructor.
   */
  Substrate() {}

  /**
   * Parametric constructor.
   *
   * @param coordinates The coordinates of substrate nodes. 
   *        Sequence is INPUT, BIAS, OUTPUT, HIDDEN.
   * @param depths The depths of substrate nodes. Same sequence with coordinates.
   * @param numInput Number of input nodes in substrate. Include one bias node.
   * @param numOutput Number of output nodes in substrate.
   * @param allowedConnectionMask This 2d array defines whether connection between
   *        node i to node j is allowed.
   * @param weightThreshold The threshold weight value to connection two nodes in substrate.
   */
  Substrate(std::vector< std::vector<double> > coordinates,
  	        std::vector<double> depths,
  	        int numInput,
  	        int numOutput,
  	        std::vector< std::vector<int> > allowedConnectionMask,
  	        double weightThreshold):
    aCoordinates(coordinates),
    aDepths(depths),
    aNumInput(numInput),
    aNumOutput(numOutput),
    aAllowedConnectionMask(allowedConnectionMask),
    aWeightThreshold(weightThreshold)
  {}

  /**
   * Destructor.
   */
  ~Substrate() {} 

  /**
   * Create genome from the substrate nodes. No link.
   *
   * @param genome The genome created from the substrate.
   */
  void InitGenome(Genome& genome)
  {
    std::vector<NeuronGene> neuronGenes;
    std::vector<LinkGene> linkGenes;

    int numNodes = aCoordinates.size();

    // Set neurons.
    for (int i = 0; i < numNodes; ++i)
    {
      if (i < aNumInput - 1)  // Input nodes.
      {
        NeuronGene inputGene(i, INPUT, LINEAR, 0, aCoordinates[i], 0, 0);
        neuronGenes.push_back(inputGene);
      }
      else if (i == aNumInput - 1)  // Bias node.
      {
        NeuronGene biasGene(i, BIAS, LINEAR, 0, aCoordinates[i], 0, 0);
        neuronGenes.push_back(biasGene);  	
      }
      else if (i >= aNumInput && i < aNumInput + aNumOutput)  // Output nodes.
      {
        NeuronGene outputGene(i, OUTPUT, SIGMOID, 1, aCoordinates[i], 0, 0);
        neuronGenes.push_back(outputGene);
      }
      else  // Hidden nodes.
      {
        NeuronGene hiddenGene(i, HIDDEN, SIGMOID, aDepths[i], aCoordinates[i], 0, 0);
        neuronGenes.push_back(hiddenGene);
      }      
    }

    // Construct genome.
    genome.Id(-1);
    genome.NumInput(aNumInput);
    genome.NumOutput(aNumOutput);
    genome.Fitness(DBL_MAX);
    genome.aNeuronGenes = neuronGenes;
    genome.aLinkGenes = linkGenes;
  }

  /**
   * Query substrate to construct network genome.
   *
   * @param cppn The genome of cppn network for link query.
   * @param genome The output genome.
   */
  void QueryLink(Genome& cppn, Genome& genome)
  {
  	// Clear links.
  	genome.aLinkGenes.clear();
    
    // Query allowed node pairs.
    int innovId = 0;
  	for (int i = 0; i < aCoordinates.size(); ++i)
  	{
  	  for (int j = 0; j < aCoordinates.size(); ++j)
  	  {
        if (aAllowedConnectionMask[i][j] != 0)
        {
          // Run cppn to query link weight.
          std::vector<double> input = genome.aNeuronGenes[i].Coordinate();
          std::vector<double> input2 = genome.aNeuronGenes[j].Coordinate();
          input.insert(input.end(), input2.begin(), input2.end());
          input.push_back(1);  // Bias.

          cppn.Activate(input);
          std::vector<double> output;
          cppn.Output(output);

          // Create new link if weight bigger than threshold.
          double weight = output[0];
          if (std::fabs(weight) > aWeightThreshold)
          {
            LinkGene link(i, j, innovId++, weight, true);  // NOTICE: we haven't scale weight.
            genome.aLinkGenes.push_back(link);
          }
        }
      }
  	}
  }

 private:

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_SUBSTRATE_HPP
