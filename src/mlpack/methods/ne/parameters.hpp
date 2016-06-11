 /**
 * @file parameters.hpp
 * @author Bang Liu
 *
 * Definition of Parameters class.
 */
#ifndef MLPACK_METHODS_NE_PARAMETERS_HPP
#define MLPACK_METHODS_NE_PARAMETERS_HPP

#include <cstddef>

#include <mlpack/core.hpp>

namespace mlpack {
namespace ne {

/**
 * This class includes different parameters for NE algorithms.
 */
class Parameters {
 public:
  // Population size.
  size_t aPopulationSize;

  // Number of generations to evolve.
  size_t aMaxGeneration;

  // Mutation rate.
  double aMutateRate;

  // Crossover rate.
  double aCrossoverRate;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_PARAMETERS_HPP