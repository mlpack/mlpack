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
  std::vector< std::vector<double> > coordinates;

  //! Depths of nodes.
  std::vector<double> depths;

  //! Number of input nodes. Including one bias node.
  int numInput;

  //! Number of output nodes.
  int numOutput;

  //! Connection allowed or not.
  //! if allowedConnectionMask[i][j] == 0, then link between node i to j is allowed.
  //! Otherwise, link not allowed. i, j is neuron id.
  std::vector< std::vector<int> > allowedConnectionMask;

  //! The threshold of weight to create new connection.
  double weightThreshold;

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
    coordinates(coordinates),
    depths(depths),
    numInput(numInput),
    numOutput(numOutput),
    allowedConnectionMask(allowedConnectionMask),
    weightThreshold(weightThreshold)
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

    int numNodes = coordinates.size();

    // Set neurons.
    for (int i = 0; i < numNodes; ++i)
    {
      if (i < numInput - 1)  // Input nodes.
      {
        NeuronGene inputGene(i, INPUT, LINEAR, 0, coordinates[i], 0, 0);
        neuronGenes.push_back(inputGene);
      }
      else if (i == numInput - 1)  // Bias node.
      {
        NeuronGene biasGene(i, BIAS, LINEAR, 0, coordinates[i], 0, 0);
        neuronGenes.push_back(biasGene);  	
      }
      else if (i >= numInput && i < numInput + numOutput)  // Output nodes.
      {
        NeuronGene outputGene(i, OUTPUT, SIGMOID, 1, coordinates[i], 0, 0);
        neuronGenes.push_back(outputGene);
      }
      else  // Hidden nodes.
      {
        NeuronGene hiddenGene(i, HIDDEN, SIGMOID, depths[i], coordinates[i], 0, 0);
        neuronGenes.push_back(hiddenGene);
      }      
    }

    // Construct genome.
    genome.Id(-1);
    genome.NumInput(numInput);
    genome.NumOutput(numOutput);
    genome.Fitness(DBL_MAX);
    genome.neuronGenes = neuronGenes;
    genome.linkGenes = linkGenes;
  }

  /**
   * Query substrate to construct network genome.
   *
   * @param cppn The genome of cppn network for link query.
   * @param genome The output genome.
   */
  void QueryLink(Genome& cppn, Genome& genome)
  {
    // assert(genome.neuronGenes.size() = coordinates.size());

  	// Clear links.
  	genome.linkGenes.clear();
    
    // Query allowed node pairs.
    int innovId = 0;
  	for (int i = 0; i < coordinates.size(); ++i)
  	{
  	  for (int j = 0; j < coordinates.size(); ++j)
  	  {
        if (allowedConnectionMask[i][j] != 0)
        {
          // Run cppn to query link weight.
          std::vector<double> input = genome.neuronGenes[i].Coordinate();
          std::vector<double> input2 = genome.neuronGenes[j].Coordinate();
          input.insert(input.end(), input2.begin(), input2.end());
          input.push_back(1);  // Bias.

          cppn.Activate(input);
          std::vector<double> output;
          cppn.Output(output);

          // Create new link if weight bigger than threshold.
          double weight = output[0];  // NOTICE: we haven't scale weight.
          if (std::fabs(weight) > weightThreshold)
          {
            LinkGene link(i, j, innovId++, weight, true);
            genome.linkGenes.push_back(link);
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
