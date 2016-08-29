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
  int speciesSize;

  //! Number of generations to evolve.
  int maxGeneration;

  //! Mutation rate.
  double mutateRate;

  //! Mutate size.
  double mutateSize;

  //! Elite percentage.
  double elitePercentage;

  //! Population size.
  int populationSize;

  //! Efficient for disjoint.
  double coeffDisjoint;

  //! Efficient for weight difference.
  double coeffWeightDiff;

  //! Threshold for judge whether belong to same species.
  double compatThreshold;

  //! Threshold for species stale age.
  int staleAgeThreshold;

  //! Crossover rate.
  double crossoverRate;

  //! Percentage to remove in each species.
  double cullSpeciesPercentage;

  //! Probability to mutate a genome's weight.
  double mutateWeightProb;

  //! Probability to mutate a genome's weight in biased way 
  //! (add Gaussian perturb noise).
  double perturbWeightProb;

  //! The Gaussian noise variance when mutating genome weights.
  double mutateWeightSize;

  //! Probability to add a forward link.
  double mutateAddForwardLinkProb;

  //! Probability to add a backward link.
  double mutateAddBackwardLinkProb;

  //! Probability to add a recurrent link.
  double mutateAddRecurrentLinkProb;

  //! Probability to add a bias link.
  double mutateAddBiasLinkProb;

  //! Probability to add neuron to genome.
  double mutateAddNeuronProb;

  //! Probability to turn enabled link to disabled.
  double mutateEnabledProb;

  //! Probability to turn disabled link to enabled.
  double mutateDisabledProb;

  //! When number of species exceed this value, start to remove stale 
  //! and weak species in population. 
  int numSpeciesThreshold;

  //! Whether the activation function of new neuron is random or not.
  bool randomTypeNewNeuron;

};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_PARAMETERS_HPP
