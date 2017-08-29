/**
 * @file cne.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Conventional Neural Evolution
 * An optimizer that works like biological evolution which selects best
 * candidates based on their fitness scores and creates new generation by
 * mutation and crossover of population.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CNE_CNE_HPP
#define MLPACK_CORE_OPTIMIZERS_CNE_CNE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Conventional Neural Evolution (CNE) is a class of evolutionary algorithms
 * focused on dealing with fixed topology. This class implements this algorithm
 * as an optimization technique to converge a given function to minima.
 *
 * The algorithm works by creating a fixed number of candidates, with random
 * weights.  Each candidate is tested upon the training set, and a fitness score
 * is assigned to it. Given the selection percentage of best candidates by the
 * user, for a single generation that many percentage of candidates are selected
 * for the next generation and the rest are removed The selected candidates for
 * a particular generation then become the parents for the next generation and
 * evolution takes place.
 *
 * The evolution process basically takes place in two types:
 * - Crossover
 * - Mutation
 *
 * Crossover takes two parents and generates two children from them. Both the
 * children have properties inherited by their parents. The parameters of the
 * parents are mixed using equal probability selection, creating two children.
 * This is just a mix of link weights or the parameters.
 *
 * In mutation parameters are updated by pertubating small noise.
 * If \f$ \Lambda \f$ is the number of weights in the network.
 *
 * \f[ \sum_{n=1}^{\Lambda}\omega_n = \omega_n + \rho, \f]
 *
 * where \f$ \rho \f$ is the small pertubation value determined randomly between
 * 0 and the mutation size given by the user as a constructor parameter. Also
 * the mutation probability taken as a constructor parameter decides the amount
 * of mutation addition into the network.
 *
 * Both the above mentioned processes create new candidates as well as change
 * the existing candidates to obtain better candidates in the next generation.
 *
 * The whole process then repeats for multiple generation until at least one of
 * the termination criteria is met:
 *
 * 1) The final value of the objective function (Not considered if not provided).
 * 2) The maximum number of generation reached (optional but highly recommended).
 * 3) Minimum change in best fitness values between two consecutive generations
 *    should be greater than a threshold value (Not considered if not provided).
 *
 * The final value and the parameters are returned by the Optimize() method.
 *
 * For CNE to work, a FunctionType template parameter is required.
 * This class must implement the following function:
 *
 *   double Evaluate(const arma::mat& iterate);
 */
class CNE
{
 public:
  /**
   * Constructor for the CNE optimizer.
   *
   * The default values provided over here are not necessarily suitable for a
   * given function. Therefore it is highly recommended to adjust the
   * parameters according to the problem.
   *
   * @param populationSize The number of candidates in the population.
   *     This should be at least 4 in size.
   * @param maxGenerations The maximum number of generations allowed for CNE.
   * @param mutationProb Probability that a weight will get mutated.
   * @param mutationSize The range of mutation noise to be added. This range
   *     is between 0 and mutationSize.
   * @param selectPercent The percentage of candidates to select to become the
   *     the next generation.
   * @param tolerance The final value of the objective function for termination.
   *     If set to negative value, tolerance is not considered.
   * @param objectiveChange Minimum change in best fitness values between two
   *     consecutive generations should be greater than threshold. If set to
   *     negative value, objectiveChange is not considered.
   */
  CNE(const size_t populationSize = 500,
      const size_t maxGenerations = 5000,
      const double mutationProb = 0.1,
      const double mutationSize = 0.02,
      const double selectPercent = 0.2,
      const double tolerance = 1e-5,
      const double objectiveChange = 1e-5);

  /**
   * Optimize the given function using CNE. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the population size.
  size_t PopulationSize() const { return populationSize; }
  //! Modify the population size.
  size_t& PopulationSize() { return populationSize; }

  //! Get maximum number of generations.
  size_t MaxGenerations() const { return maxGenerations; }
  //! Modify maximum number of generations.
  size_t& MaxGenerations() { return maxGenerations; }

  //! Get the mutation probability.
  double MutationProbability() const { return mutationProb; }
  //! Modify the mutation probability.
  double& MutationProbability() { return mutationProb; }

  //! Get the mutation size.
  double MutationSize() const { return mutationSize; }
  //! Modify the mutation size.
  double& MutationSize() { return mutationSize; }

  //! Get the selection percentage.
  double SelectionPercentage() const { return selectPercent; }
  //! Modify the selection percentage.
  double& SelectionPercentage() { return selectPercent; }

  //! Get the final objective value.
  double Tolerance() const { return tolerance; }
  //! Modify the final objective value.
  double& Tolerance() { return tolerance; }

  //! Get the change in fitness history between generations.
  double ObjectiveChange() const { return objectiveChange; }
  //! Modify the termination criteria of change in fitness value.
  double& ObjectiveChange() { return objectiveChange; }

 private:
  //! Reproduce candidates to create the next generation.
  void Reproduce();

  //! Modify weights with some noise for the evolution of next generation.
  void Mutate();

  /**
   * Crossover parents and create new childs. Two parents create two new childs.
   *
   * @param mom First parent from the elite population.
   * @param dad Second parent from the elite population.
   * @param dropout1 The place to delete the candidate of the present
   *                 generation and place a child over there for the
   *                 next generation.
   * @param dropout2 The place to delete the candidate of the present
   *                 generation and place a child over there for the
   *                 next generation.
   */
  void Crossover(const size_t mom,
                 const size_t dad,
                 const size_t dropout1,
                 const size_t dropout2);

  //! Population matrix. Each column is a candidate.
  arma::cube population;

  //! Vector of fintness values corresponding to each candidate.
  arma::vec fitnessValues;

  //! Index of sorted fitness values.
  arma::uvec index;

  //! The number of candidates in the population.
  size_t populationSize;

  //! Maximum number of generations before termination criteria is met.
  size_t maxGenerations;

  //! Probability that a weight will get mutated.
  double mutationProb;

  //! The range of mutation noise to be added.
  double mutationSize;

  //! The percentage of best candidates to be selected for parents.
  double selectPercent;

  //! The final value of the objective function.
  double tolerance;

  //! Minimum change in best fitness values between two generations.
  double objectiveChange;

  //! Number of candidates to become parent for the next generation.
  size_t numElite;

  //! Store the number of elements in a cube slice or a matrix column.
  size_t elements;
};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cne_impl.hpp"

#endif
