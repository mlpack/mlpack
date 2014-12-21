/**
 * @file lbfgs.hpp
 * @author Dongryeol Lee
 * @author Ryan Curtin
 *
 * The generic L-BFGS optimizer.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP
#define __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * The generic L-BFGS optimizer, which uses a back-tracking line search
 * algorithm to minimize a function.  The parameters for the algorithm (number
 * of memory points, maximum step size, and so forth) are all configurable via
 * either the constructor or standalone modifier functions.  A function which
 * can be optimized by this class must implement the following methods:
 *
 *  - a default constructor
 *  - double Evaluate(const arma::mat& coordinates);
 *  - void Gradient(const arma::mat& coordinates, arma::mat& gradient);
 *  - arma::mat& GetInitialPoint();
 */
template<typename FunctionType>
class L_BFGS
{
 public:
  /**
   * Initialize the L-BFGS object.  Store a reference to the function we will be
   * optimizing and set the size of the memory for the algorithm.  There are
   * many parameters that can be set for the optimization, but default values
   * are given for each of them.
   *
   * @param function Instance of function to be optimized.
   * @param numBasis Number of memory points to be stored (default 5).
   * @param maxIterations Maximum number of iterations for the optimization
   *     (default 0 -- may run indefinitely).
   * @param armijoConstant Controls the accuracy of the line search routine for
   *     determining the Armijo condition.
   * @param wolfe Parameter for detecting the Wolfe condition.
   * @param minGradientNorm Minimum gradient norm required to continue the
   *     optimization.
   * @param maxLineSearchTrials The maximum number of trials for the line search
   *     (before giving up).
   * @param minStep The minimum step of the line search.
   * @param maxStep The maximum step of the line search.
   */
  L_BFGS(FunctionType& function,
         const size_t numBasis = 5, /* entirely arbitrary */
         const size_t maxIterations = 0, /* run forever */
         const double armijoConstant = 1e-4,
         const double wolfe = 0.9,
         const double minGradientNorm = 1e-10,
         const size_t maxLineSearchTrials = 50,
         const double minStep = 1e-20,
         const double maxStep = 1e20);

  /**
   * Return the point where the lowest function value has been found.
   *
   * @return arma::vec representing the point and a double with the function
   *     value at that point.
   */
  const std::pair<arma::mat, double>& MinPointIterate() const;

  /**
   * Use L-BFGS to optimize the given function, starting at the given iterate
   * point and finding the minimum.  The maximum number of iterations is set in
   * the constructor (or with MaxIterations()).  Alternately, another overload
   * is provided which takes a maximum number of iterations as a parameter.  The
   * given starting point will be modified to store the finishing point of the
   * algorithm, and the final objective value is returned.
   *
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate);

  /**
   * Use L-BFGS to optimize (minimize) the given function, starting at the given
   * iterate point, and performing no more than the given maximum number of
   * iterations (the class variable maxIterations is ignored for this run, but
   * not modified).  The given starting point will be modified to store the
   * finishing point of the algorithm, and the final objective value is
   * returned.
   *
   * @param iterate Starting point (will be modified).
   * @param maxIterations Maximum number of iterations (0 specifies no limit).
   * @return Objective value of the final point.
   */
  double Optimize(arma::mat& iterate, const size_t maxIterations);

  //! Return the function that is being optimized.
  const FunctionType& Function() const { return function; }
  //! Modify the function that is being optimized.
  FunctionType& Function() { return function; }

  //! Get the memory size.
  size_t NumBasis() const { return numBasis; }
  //! Modify the memory size.
  size_t& NumBasis() { return numBasis; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the Armijo condition constant.
  double ArmijoConstant() const { return armijoConstant; }
  //! Modify the Armijo condition constant.
  double& ArmijoConstant() { return armijoConstant; }

  //! Get the Wolfe parameter.
  double Wolfe() const { return wolfe; }
  //! Modify the Wolfe parameter.
  double& Wolfe() { return wolfe; }

  //! Get the minimum gradient norm.
  double MinGradientNorm() const { return minGradientNorm; }
  //! Modify the minimum gradient norm.
  double& MinGradientNorm() { return minGradientNorm; }

  //! Get the maximum number of line search trials.
  size_t MaxLineSearchTrials() const { return maxLineSearchTrials; }
  //! Modify the maximum number of line search trials.
  size_t& MaxLineSearchTrials() { return maxLineSearchTrials; }

  //! Return the minimum line search step size.
  double MinStep() const { return minStep; }
  //! Modify the minimum line search step size.
  double& MinStep() { return minStep; }

  //! Return the maximum line search step size.
  double MaxStep() const { return maxStep; }
  //! Modify the maximum line search step size.
  double& MaxStep() { return maxStep; }

  // convert the obkect into a string
  std::string ToString() const;

 private:
  //! Internal reference to the function we are optimizing.
  FunctionType& function;

  //! Position of the new iterate.
  arma::mat newIterateTmp;
  //! Stores all the s matrices in memory.
  arma::cube s;
  //! Stores all the y matrices in memory.
  arma::cube y;

  //! Size of memory for this L-BFGS optimizer.
  size_t numBasis;
  //! Maximum number of iterations.
  size_t maxIterations;
  //! Parameter for determining the Armijo condition.
  double armijoConstant;
  //! Parameter for detecting the Wolfe condition.
  double wolfe;
  //! Minimum gradient norm required to continue the optimization.
  double minGradientNorm;
  //! Maximum number of trials for the line search.
  size_t maxLineSearchTrials;
  //! Minimum step of the line search.
  double minStep;
  //! Maximum step of the line search.
  double maxStep;

  //! Best point found so far.
  std::pair<arma::mat, double> minPointIterate;

  /**
   * Evaluate the function at the given iterate point and store the result if it
   * is a new minimum.
   *
   * @return The value of the function.
   */
  double Evaluate(const arma::mat& iterate);

  /**
   * Calculate the scaling factor, gamma, which is used to scale the Hessian
   * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal
   * (1989).
   *
   * @return The calculated scaling factor.
   */
  double ChooseScalingFactor(const size_t iterationNum,
                             const arma::mat& gradient);

  /**
   * Check to make sure that the norm of the gradient is not smaller than 1e-5.
   * Currently that value is not configurable.
   *
   * @return (norm < minGradientNorm).
   */
  bool GradientNormTooSmall(const arma::mat& gradient);

  /**
   * Perform a back-tracking line search along the search direction to
   * calculate a step size satisfying the Wolfe conditions.  The parameter
   * iterate will be modified if the method is successful.
   *
   * @param functionValue Value of the function at the initial point
   * @param iterate The initial point to begin the line search from
   * @param gradient The gradient at the initial point
   * @param searchDirection A vector specifying the search direction
   * @param stepSize Variable the calculated step size will be stored in
   *
   * @return false if no step size is suitable, true otherwise.
   */
  bool LineSearch(double& functionValue,
                  arma::mat& iterate,
                  arma::mat& gradient,
                  const arma::mat& searchDirection);

  /**
   * Find the L-BFGS search direction.
   *
   * @param gradient The gradient at the current point
   * @param iteration_num The iteration number
   * @param scaling_factor Scaling factor to use (see ChooseScalingFactor_())
   * @param search_direction Vector to store search direction in
   */
  void SearchDirection(const arma::mat& gradient,
                       const size_t iterationNum,
                       const double scalingFactor,
                       arma::mat& searchDirection);

  /**
   * Update the y and s matrices, which store the differences
   * between the iterate and old iterate and the differences between the
   * gradient and the old gradient, respectively.
   *
   * @param iterationNum Iteration number
   * @param iterate Current point
   * @param oldIterate Point at last iteration
   * @param gradient Gradient at current point (iterate)
   * @param oldGradient Gradient at last iteration point (oldIterate)
   */
  void UpdateBasisSet(const size_t iterationNum,
                      const arma::mat& iterate,
                      const arma::mat& oldIterate,
                      const arma::mat& gradient,
                      const arma::mat& oldGradient);
};

}; // namespace optimization
}; // namespace mlpack

#include "lbfgs_impl.hpp"

#endif // __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP

