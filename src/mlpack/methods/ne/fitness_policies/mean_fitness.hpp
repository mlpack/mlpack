/**
 * @file mean_fitness.hpp
 * @author Marcus Edel
 *
 * This very simple policy calculates the mean value over all fitnesses.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FITNESS_POLICIES_MEAN_FITNESS_HPP
#define MLPACK_METHODS_FITNESS_POLICIES_MEAN_FITNESS_HPP

#include <mlpack/core.hpp>

#include "../species.hpp"

namespace mlpack {
namespace ne {

/**
 * Policy which calculates the mean value over all fitnesses.
 */
class MeanFitness
{
 public:
  //! Default constructor required by MeanFitness policy.
  MeanFitness() { /* Nothing to do here */ }

  static inline force_inline double Fitness(const Species& species)
  {
    double fitness = 0;
    for (const Genome& genome : species.genomes)
    {
      fitness += genome.Fitness();
    }

    return fitness / species.Size();
  }

  //! Serialize the mean fitness policy (nothing to do).
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */)
  {
    // Nothing to do here.
  }
};

} // namespace ne
} // namespace mlpack

#endif