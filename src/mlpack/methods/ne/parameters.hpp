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
  // Species size.
  ssize_t aSpeciesSize;

  // Number of generations to evolve.
  ssize_t aMaxGeneration;

  // Mutation rate.
  double aMutateRate;

  // Mutate size.
  double aMutateSize;

  // Elite percentage.
  double aElitePercentage;

  // Population size.
  ssize_t aPopulationSize;

  // Efficient for disjoint.
  double aCoeffDisjoint;

  // Efficient for weight difference.
  double aCoeffWeightDiff;

  // Threshold for judge whether belong to same species.
  double aCompatThreshold;

  // Threshold for species stale age.
  ssize_t aStaleAgeThreshold;

  // Crossover rate.
  double aCrossoverRate;

  // Percentage to remove in each species.
  double aCullSpeciesPercentage;

  // Probability to mutate a genome's weight
  double aMutateWeightProb;

  // Probability to mutate a genome's weight in biased way (add Gaussian perturb noise).
  double aPerturbWeightProb;

  // The Gaussian noise variance when mutating genome weights.
  double aMutateWeightSize;

  // Probability to add link to genome.
  double aMutateAddLinkProb;

  // Probability to add neuron to genome.
  double aMutateAddNeuronProb;

  // Probability to turn enabled link to disabled.
  double aMutateEnabledProb;

  // Probability to turn disabled link to enabled.
  double aMutateDisabledProb;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_PARAMETERS_HPP