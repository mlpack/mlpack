 /**
 * @file parameters.hpp
 * @author Bang Liu
 *
 * Definition of Parameters class.
 */
#ifndef MLPACK_METHODS_NE_PARAMETERS_HPP
#define MLPACK_METHODS_NE_PARAMETERS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ne {

/**
 * This class includes different parameters for NE algorithms.
 */
class Parameters
{
 public:
  //! Species size.
  int aSpeciesSize;

  //! Number of generations to evolve.
  int aMaxGeneration;

  //! Mutation rate.
  double aMutateRate;

  //! Mutate size.
  double aMutateSize;

  //! Elite percentage.
  double aElitePercentage;

  //! Population size.
  int aPopulationSize;

  //! Efficient for disjoint.
  double aCoeffDisjoint;

  //! Efficient for weight difference.
  double aCoeffWeightDiff;

  //! Threshold for judge whether belong to same species.
  double aCompatThreshold;

  //! Threshold for species stale age.
  int aStaleAgeThreshold;

  //! Crossover rate.
  double aCrossoverRate;

  //! Percentage to remove in each species.
  double aCullSpeciesPercentage;

  //! Probability to mutate a genome's weight.
  double aMutateWeightProb;

  //! Probability to mutate a genome's weight in biased way (add Gaussian perturb noise).
  double aPerturbWeightProb;

  //! The Gaussian noise variance when mutating genome weights.
  double aMutateWeightSize;

  //! Probability to add a forward link.
  double aMutateAddForwardLinkProb;

  //! Probability to add a backward link.
  double aMutateAddBackwardLinkProb;

  //! Probability to add a recurrent link.
  double aMutateAddRecurrentLinkProb;

  //! Probability to add a bias link.
  double aMutateAddBiasLinkProb;

  //! Probability to add neuron to genome.
  double aMutateAddNeuronProb;

  //! Probability to turn enabled link to disabled.
  double aMutateEnabledProb;

  //! Probability to turn disabled link to enabled.
  double aMutateDisabledProb;

  //! When number of species exceed this value, start to remove stale and weak species in population. 
  int aNumSpeciesThreshold;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_PARAMETERS_HPP