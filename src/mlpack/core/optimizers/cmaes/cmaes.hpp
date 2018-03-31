/**
 * @file cmaes.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Definition of the Covariance Matrix Adaptation Evolution Strategy as proposed
 * by N. Hansen et al. in "Completely Derandomized Self-Adaptation in Evolution
 * Strategies".
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_HPP

#include <mlpack/prereqs.hpp>

#include "full_selection.hpp"
#include "random_selection.hpp"

namespace mlpack {
namespace optimization {

/**
 * CMA-ES - Covariance Matrix Adaptation Evolution Strategy is s a stochastic
 * search algorithm. CMA-ES is a second order approach estimating a positive
 * definite matrix within an iterative procedure using the covariance matrix.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Hansen2001
 *   author    = {Hansen, Nikolaus and Ostermeier, Andreas},
 *   title     = {Completely Derandomized Self-Adaptation in Evolution
 *                Strategies},
 *   journal   = {Evol. Comput.},
 *   volume    = {9},
 *   number    = {2},
 *   year      = {2001},
 *   pages     = {159--195},
 *   publisher = {MIT Press},
 * }
 * @endcode
 *
 * For CMA-ES to work, the class must implement the following function:
 *
 *   size_t NumFunctions();
 *   double Evaluate(const arma::mat& coordinates, const size_t i);
 *   void Gradient(const arma::mat& coordinates,
 *                 const size_t i,
 *                 arma::mat& gradient);
 *
 * NumFunctions() should return the number of functions (\f$n\f$), and in the
 * other two functions, the parameter i refers to which individual function (or
 * gradient) is being evaluated.  So, for the case of a data-dependent function,
 * such as NCA (see mlpack::nca::NCA), NumFunctions() should return the number
 * of points in the dataset, and Evaluate(coordinates, 0) will evaluate the
 * objective function on the first point in the dataset (presumably, the dataset
 * is held internally in the DecomposableFunctionType).
 *
 * @tparam SelectionPolicy The selection strategy used for the evaluation step.
 */
template<typename SelectionPolicyType = FullSelection>
class CMAES
{
 public:
  /**
   * Construct the CMA-ES optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param lambda The population size (0 use the default size).
   * @param lowerBound Lower bound of decision variables.
   * @param upperBound Upper bound of decision variables.
   * @param batchSize Batch size to use for the objective calculation.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param selectionPolicy Instantiated selection policy used to calculate the
   *     objective.
   */
  CMAES(const size_t lambda = 0,
        const double lowerBound = -10,
        const double upperBound = 10,
        const size_t batchSize = 32,
        const size_t maxIterations = 1000,
        const double tolerance = 1e-5,
        const SelectionPolicyType& selectionPolicy = SelectionPolicyType());

  /**
   * Optimize the given function using CMA-ES. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the step size.
  size_t PopulationSize() const { return lambda; }
  //! Modify the step size.
  size_t& PopulationSize() { return lambda; }

  //! Get the lower bound of decision variables.
  double LowerBound() const { return lowerBound; }
  //! Modify the lower bound of decision variables.
  double& LowerBound() { return lowerBound; }

  //! Get the upper bound of decision variables
  double UpperBound() const { return upperBound; }
  //! Modify the upper bound of decision variables
  double& UpperBound() { return upperBound; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get the selection policy.
  const SelectionPolicyType& SelectionPolicy() const { return selectionPolicy; }
  //! Modify the selection policy.
  SelectionPolicyType& SelectionPolicy() { return selectionPolicy; }

 private:
  //! Population size.
  size_t lambda;

  //! Lower bound of decision variables.
  double lowerBound;

  //! Upper bound of decision variables
  double upperBound;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The selection policy used to calculate the objective.
  SelectionPolicyType selectionPolicy;
};

/**
 * Convenient typedef for CMAES approximation.
 */
template<typename SelectionPolicyType = RandomSelection>
using ApproxCMAES = CMAES<SelectionPolicyType>;

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif
