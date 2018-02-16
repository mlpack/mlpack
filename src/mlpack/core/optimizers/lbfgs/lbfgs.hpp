/**
 * @file lbfgs.hpp
 * @author Dongryeol Lee
 * @author Ryan Curtin
 *
 * The generic L-BFGS optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP
#define MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

/**
 * The generic L-BFGS optimizer, which uses a back-tracking line search
 * algorithm to minimize a function.  The parameters for the algorithm (number
 * of memory points, maximum step size, and so forth) are all configurable via
 * either the constructor or standalone modifier functions.  A function which
 * can be optimized by this class's Optimize() method must implement the
 * following methods:
 *
 *  - a default constructor
 *  - double Evaluate(const arma::mat& coordinates);
 *  - void Gradient(const arma::mat& coordinates, arma::mat& gradient);
 *  - arma::mat& GetInitialPoint();
 */
class L_BFGS
{
 public:
  /**
   * Initialize the L-BFGS object.  There are many parameters that can be set
   * for the optimization, but default values are given for each of them.
   *
   * @param numBasis Number of memory points to be stored (default 5).
   * @param maxIterations Maximum number of iterations for the optimization
   *     (0 means no limit and may run indefinitely).
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
  L_BFGS(const size_t numBasis = 10, /* same default as scipy */
         const size_t maxIterations = 10000, /* many but not infinite */
         const double armijoConstant = 1e-4,
         const double wolfe = 0.9,
         const double minGradientNorm = 1e-6,
         const double factr = 1e-15,
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
   * @tparam FunctionType Type of the function to be optimized.
   * @param function Function to optimize; must have Evaluate() and Gradient().
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename FunctionType>
  double Optimize(FunctionType& function, arma::mat& iterate);

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

  //! Get the factr value.
  double Factr() const { return factr; }
  //! Modify the factr value.
  double& Factr() { return factr; }

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

 private:
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
  //! Minimum relative function value decrease to continue the optimization.
  double factr;
  //! Maximum number of trials for the line search.
  size_t maxLineSearchTrials;
  //! Minimum step of the line search.
  double minStep;
  //! Maximum step of the line search.
  double maxStep;

  /**
   * Calculate the scaling factor, gamma, which is used to scale the Hessian
   * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal
   * (1989).
   *
   * @return The calculated scaling factor.
   */
  double ChooseScalingFactor(const size_t iterationNum,
                             const arma::mat& gradient,
                             const arma::cube& s,
                             const arma::cube& y);

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
  template<typename FunctionType>
  bool LineSearch(FunctionType& function,
                  double& functionValue,
                  arma::mat& iterate,
                  arma::mat& gradient,
                  arma::mat& newIterateTmp,
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
                       const arma::cube& s,
                       const arma::cube& y,
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
                      const arma::mat& oldGradient,
                      arma::cube& s,
                      arma::cube& y);
};

} // namespace optimization
} // namespace mlpack

#include "lbfgs_impl.hpp"

#endif // MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP

