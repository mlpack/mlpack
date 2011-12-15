/**
 * @file lbfgs.hpp
 * @author Dongryeol Lee
 * @author Ryan Curtin
 *
 * The generic L-BFGS optimizer.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP
#define __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

template<typename FunctionType>
class L_BFGS
{
 public:
  /**
   * Initialize the L-BFGS object.  Copy the function we will be optimizing
   * and set the size of the memory for the algorithm.
   *
   * @param function Instance of function to be optimized
   * @param numBasis Number of memory points to be stored
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
  L_BFGS(const FunctionType& function,
         const size_t numBasis,
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
   * point and performing no more than the specified number of maximum
   * iterations.  The given starting point will be modified to store the
   * finishing point of the algorithm.
   *
   * @param maxIterations Maximum number of iterations to perform
   * @param iterate Starting point (will be modified)
   */
  bool Optimize(const size_t maxIterations, arma::mat& iterate);

  //! Get the memory size.
  size_t NumBasis() const { return numBasis; }
  //! Modify the memory size.
  size_t& NumBasis() { return numBasis; }

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

 private:
  //! Internal copy of the function we are optimizing.
  FunctionType function;

  //! Position of the new iterate.
  arma::mat newIterateTmp;
  //! Stores all the s matrices in memory.
  arma::cube s;
  //! Stores all the y matrices in memory.
  arma::cube y;

  //! Size of memory for this L-BFGS optimizer.
  size_t numBasis;
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
   * @return The value of the function
   */
  double Evaluate(const arma::mat& iterate);

  /**
   * Calculate the scaling factor gamma which is used to scale the Hessian
   * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal
   * (1989).
   *
   * @return The calculated scaling factor
   */
  double ChooseScalingFactor(const size_t iterationNum,
                             const arma::mat& gradient);

  /**
   * Check to make sure that the norm of the gradient is not smaller than 1e-5.
   * Currently that value is not configurable.
   *
   * @return (norm < minGradientNorm)
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
                  const arma::mat& searchDirection,
                  double& stepSize);

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
