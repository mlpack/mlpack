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
class Substrate {
 public:
  std::vector<std::vector<double>> aNodeCoordinates;
  Genome aGenome;


 private:



};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_SUBSTRATE_HPP
