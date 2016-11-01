/**
 * @file lbfgs_impl.hpp
 * @author Dongryeol Lee (dongryel@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * The implementation of the L_BFGS optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP

namespace mlpack {
namespace optimization {

/**
 * Initialize the L_BFGS object.  Copy the function we will be optimizing and
 * set the size of the memory for the algorithm.
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
template<typename FunctionType>
L_BFGS<FunctionType>::L_BFGS(FunctionType& function,
                             const size_t numBasis,
                             const size_t maxIterations,
                             const double armijoConstant,
                             const double wolfe,
                             const double minGradientNorm,
                             const double factr,
                             const size_t maxLineSearchTrials,
                             const double minStep,
                             const double maxStep) :
    function(function),
    numBasis(numBasis),
    maxIterations(maxIterations),
    armijoConstant(armijoConstant),
    wolfe(wolfe),
    minGradientNorm(minGradientNorm),
    factr(factr),
    maxLineSearchTrials(maxLineSearchTrials),
    minStep(minStep),
    maxStep(maxStep)
{
  // Get the dimensions of the coordinates of the function; GetInitialPoint()
  // might return an arma::vec, but that's okay because then n_cols will simply
  // be 1.
  const size_t rows = function.GetInitialPoint().n_rows;
  const size_t cols = function.GetInitialPoint().n_cols;

  newIterateTmp.set_size(rows, cols);
  s.set_size(rows, cols, numBasis);
  y.set_size(rows, cols, numBasis);

  // Allocate the pair holding the min iterate information.
  minPointIterate.first.zeros(rows, cols);
  minPointIterate.second = std::numeric_limits<double>::max();
}

/**
 * Evaluate the function at the given iterate point and store the result if
 * it is a new minimum.
 *
 * @return The value of the function
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::Evaluate(const arma::mat& iterate)
{
  // Evaluate the function and keep track of the minimum function
  // value encountered during the optimization.
  double functionValue = function.Evaluate(iterate);

  if (functionValue < minPointIterate.second)
  {
    minPointIterate.first = iterate;
    minPointIterate.second = functionValue;
  }

  return functionValue;
}

/**
 * Calculate the scaling factor gamma which is used to scale the Hessian
 * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal (1989).
 *
 * @return The calculated scaling factor
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::ChooseScalingFactor(const size_t iterationNum,
                                                 const arma::mat& gradient)
{
  double scalingFactor = 1.0;
  if (iterationNum > 0)
  {
    int previousPos = (iterationNum - 1) % numBasis;
    // Get s and y matrices once instead of multiple times.
    arma::mat& sMat = s.slice(previousPos);
    arma::mat& yMat = y.slice(previousPos);
    scalingFactor = dot(sMat, yMat) / dot(yMat, yMat);
  }
  else
  {
    scalingFactor = 1.0 / sqrt(dot(gradient, gradient));
  }

  return scalingFactor;
}

/**
 * Check to make sure that the norm of the gradient is not smaller than 1e-10.
 * Currently that value is not configurable.
 *
 * @return (norm < minGradientNorm)
 */
template<typename FunctionType>
bool L_BFGS<FunctionType>::GradientNormTooSmall(const arma::mat& gradient)
{
  double norm = arma::norm(gradient, 2);

  return (norm < minGradientNorm);
}

/**
 * Perform a back-tracking line search along the search direction to calculate a
 * step size satisfying the Wolfe conditions.
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
bool L_BFGS<FunctionType>::LineSearch(double& functionValue,
                                      arma::mat& iterate,
                                      arma::mat& gradient,
                                      const arma::mat& searchDirection)
{
  // Default first step size of 1.0.
  double stepSize = 1.0;

  // The initial linear term approximation in the direction of the
  // search direction.
  double initialSearchDirectionDotGradient =
      arma::dot(gradient, searchDirection);

  // If it is not a descent direction, just report failure.
  if (initialSearchDirectionDotGradient > 0.0)
  {
    Log::Warn << "L-BFGS line search direction is not a descent direction "
        << "(terminating)!" << std::endl;
    return false;
  }

  // Save the initial function value.
  double initialFunctionValue = functionValue;

  // Unit linear approximation to the decrease in function value.
  double linearApproxFunctionValueDecrease = armijoConstant *
      initialSearchDirectionDotGradient;

  // The number of iteration in the search.
  size_t numIterations = 0;

  // Armijo step size scaling factor for increase and decrease.
  const double inc = 2.1;
  const double dec = 0.5;
  double width = 0;

  while (true)
  {
    // Perform a step and evaluate the gradient and the function values at that
    // point.
    newIterateTmp = iterate;
    newIterateTmp += stepSize * searchDirection;
    functionValue = Evaluate(newIterateTmp);
    function.Gradient(newIterateTmp, gradient);
    numIterations++;

    if (functionValue > initialFunctionValue + stepSize *
        linearApproxFunctionValueDecrease)
    {
      width = dec;
    }
    else
    {
      // Check Wolfe's condition.
      double searchDirectionDotGradient = arma::dot(gradient, searchDirection);

      if (searchDirectionDotGradient < wolfe *
          initialSearchDirectionDotGradient)
      {
        width = inc;
      }
      else
      {
        if (searchDirectionDotGradient > -wolfe *
            initialSearchDirectionDotGradient)
        {
          width = dec;
        }
        else
        {
          break;
        }
      }
    }

    // Terminate when the step size gets too small or too big or it
    // exceeds the max number of iterations.
    const bool cond1 = (stepSize < minStep);
    const bool cond2 = (stepSize > maxStep);
    const bool cond3 = (numIterations >= maxLineSearchTrials);
    if (cond1 || cond2 || cond3)
      break;

    // Scale the step size.
    stepSize *= width;
  }

  // Move to the new iterate.
  iterate = newIterateTmp;
  return true;
}

/**
 * Find the L_BFGS search direction.
 *
 * @param gradient The gradient at the current point
 * @param iterationNum The iteration number
 * @param scalingFactor Scaling factor to use (see ChooseScalingFactor_())
 * @param searchDirection Vector to store search direction in
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::SearchDirection(const arma::mat& gradient,
                                           const size_t iterationNum,
                                           const double scalingFactor,
                                           arma::mat& searchDirection)
{
  // Start from this point.
  searchDirection = gradient;

  // See "A Recursive Formula to Compute H * g" in "Updating quasi-Newton
  // matrices with limited storage" (Nocedal, 1980).

  // Temporary variables.
  arma::vec rho(numBasis);
  arma::vec alpha(numBasis);

  size_t limit = (numBasis > iterationNum) ? 0 : (iterationNum - numBasis);
  for (size_t i = iterationNum; i != limit; i--)
  {
    int translatedPosition = (i + (numBasis - 1)) % numBasis;
    rho[iterationNum - i] = 1.0 / arma::dot(y.slice(translatedPosition),
                                            s.slice(translatedPosition));
    alpha[iterationNum - i] = rho[iterationNum - i] *
        arma::dot(s.slice(translatedPosition), searchDirection);
    searchDirection -= alpha[iterationNum - i] * y.slice(translatedPosition);
  }

  searchDirection *= scalingFactor;

  for (size_t i = limit; i < iterationNum; i++)
  {
    int translatedPosition = i % numBasis;
    double beta = rho[iterationNum - i - 1] *
        arma::dot(y.slice(translatedPosition), searchDirection);
    searchDirection += (alpha[iterationNum - i - 1] - beta) *
        s.slice(translatedPosition);
  }

  // Negate the search direction so that it is a descent direction.
  searchDirection *= -1;
}

/**
 * Update the y and s matrices, which store the differences between
 * the iterate and old iterate and the differences between the gradient and the
 * old gradient, respectively.
 *
 * @param iterationNum Iteration number
 * @param iterate Current point
 * @param oldIterate Point at last iteration
 * @param gradient Gradient at current point (iterate)
 * @param oldGradient Gradient at last iteration point (oldIterate)
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::UpdateBasisSet(const size_t iterationNum,
                                          const arma::mat& iterate,
                                          const arma::mat& oldIterate,
                                          const arma::mat& gradient,
                                          const arma::mat& oldGradient)
{
  // Overwrite a certain position instead of pushing everything in the vector
  // back one position.
  int overwritePos = iterationNum % numBasis;
  s.slice(overwritePos) = iterate - oldIterate;
  y.slice(overwritePos) = gradient - oldGradient;
}

/**
 * Return the point where the lowest function value has been found.
 *
 * @return arma::vec representing the point and a double with the function
 *     value at that point.
 */
template<typename FunctionType>
inline const std::pair<arma::mat, double>&
L_BFGS<FunctionType>::MinPointIterate() const
{
  return minPointIterate;
}

template<typename FunctionType>
inline double L_BFGS<FunctionType>::Optimize(arma::mat& iterate)
{
  return Optimize(iterate, maxIterations);
}

/**
 * Use L_BFGS to optimize the given function, starting at the given iterate
 * point and performing no more than the specified number of maximum iterations.
 * The given starting point will be modified to store the finishing point of the
 * algorithm.
 *
 * @param numIterations Maximum number of iterations to perform
 * @param iterate Starting point (will be modified)
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::Optimize(arma::mat& iterate,
                                      const size_t maxIterations)
{
  // Ensure that the cubes holding past iterations' information are the right
  // size.  Also set the current best point value to the maximum.
  const size_t rows = function.GetInitialPoint().n_rows;
  const size_t cols = function.GetInitialPoint().n_cols;

  s.set_size(rows, cols, numBasis);
  y.set_size(rows, cols, numBasis);
  minPointIterate.second = std::numeric_limits<double>::max();

  // The old iterate to be saved.
  arma::mat oldIterate;
  oldIterate.zeros(iterate.n_rows, iterate.n_cols);

  // Whether to optimize until convergence.
  bool optimizeUntilConvergence = (maxIterations == 0);

  // The initial function value.
  double functionValue = Evaluate(iterate);
  double prevFunctionValue = functionValue;

  // The gradient: the current and the old.
  arma::mat gradient;
  arma::mat oldGradient;
  gradient.zeros(iterate.n_rows, iterate.n_cols);
  oldGradient.zeros(iterate.n_rows, iterate.n_cols);

  // The search direction.
  arma::mat searchDirection;
  searchDirection.zeros(iterate.n_rows, iterate.n_cols);

  // The initial gradient value.
  function.Gradient(iterate, gradient);

  // The main optimization loop.
  for (size_t itNum = 0; optimizeUntilConvergence || (itNum != maxIterations);
       ++itNum)
  {
    Log::Debug << "L-BFGS iteration " << itNum << "; objective " <<
        function.Evaluate(iterate) << ", gradient norm " <<
        arma::norm(gradient, 2) << ", " <<
        ((prevFunctionValue - functionValue) /
         std::max(std::max(fabs(prevFunctionValue), fabs(functionValue)), 1.0)) << "." << std::endl;

    prevFunctionValue = functionValue;

    // Break when the norm of the gradient becomes too small.
    //
    // But don't do this on the first iteration to ensure we always take at
    // least one descent step.
    if (itNum > 0 && GradientNormTooSmall(gradient))
    {
      Log::Debug << "L-BFGS gradient norm too small (terminating successfully)."
          << std::endl;
      break;
    }

    // Break if the objective is not a number.
    if (std::isnan(functionValue))
    {
      Log::Warn << "L-BFGS terminated with objective " << functionValue << "; "
          << "are the objective and gradient functions implemented correctly?"
          << std::endl;
      break;
    }

    // Choose the scaling factor.
    double scalingFactor = ChooseScalingFactor(itNum, gradient);

    // Build an approximation to the Hessian and choose the search
    // direction for the current iteration.
    SearchDirection(gradient, itNum, scalingFactor, searchDirection);

    // Save the old iterate and the gradient before stepping.
    oldIterate = iterate;
    oldGradient = gradient;

    // Do a line search and take a step.
    if (!LineSearch(functionValue, iterate, gradient, searchDirection))
    {
      Log::Debug << "Line search failed.  Stopping optimization." << std::endl;
      break; // The line search failed; nothing else to try.
    }

    // It is possible that the difference between the two coordinates is zero.
    // In this case we terminate successfully.
    if (accu(iterate != oldIterate) == 0)
    {
      Log::Debug << "L-BFGS step size of 0 (terminating successfully)."
          << std::endl;
      break;
    }

    // If we can't make progress on the gradient, then we'll also accept
    // a stable function value.
    const double denom =
      std::max(
        std::max(fabs(prevFunctionValue), fabs(functionValue)),
        1.0);
    if ((prevFunctionValue - functionValue) / denom <= factr)
    {
      Log::Debug << "L-BFGS function value stable (terminating successfully)."
          << std::endl;
      break;
    }

    // Overwrite an old basis set.
    UpdateBasisSet(itNum, iterate, oldIterate, gradient, oldGradient);

  } // End of the optimization loop.

  return function.Evaluate(iterate);
}

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP

