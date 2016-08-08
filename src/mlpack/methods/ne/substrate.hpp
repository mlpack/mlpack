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
  //! Coordinates of nodes.
  std::vector<std::vector<double>> aNodeCoordinates;

  //! Genome created from the substrate nodes.
  Genome aGenome;


 private:

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_SUBSTRATE_HPP
