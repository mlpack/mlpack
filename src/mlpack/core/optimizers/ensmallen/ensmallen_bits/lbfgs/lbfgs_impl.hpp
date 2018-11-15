/**
 * @file lbfgs_impl.hpp
 * @author Dongryeol Lee (dongryel@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * The implementation of the L_BFGS optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LBFGS_LBFGS_IMPL_HPP
#define ENSMALLEN_LBFGS_LBFGS_IMPL_HPP

// In case it hasn't been included yet.
#include "lbfgs.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

/**
 * Initialize the L_BFGS object.
 *
 * @param numBasis Number of memory points to be stored (default 5).
 * @param maxIterations Maximum number of iterations for the optimization
 *     (0 means no limit and may run indefinitely).
 * @param armijoConstant Controls the accuracy of the line search routine for
 *     determining the Armijo condition.
 * @param wolfe Parameter for detecting the Wolfe condition.
 * @param minGradientNorm Minimum gradient norm required to continue the
 *     optimization.
 * @param factr Minimum relative function value decrease to continue
 *     the optimization.
 * @param maxLineSearchTrials The maximum number of trials for the line search
 *     (before giving up).
 * @param minStep The minimum step of the line search.
 * @param maxStep The maximum step of the line search.
 */
inline L_BFGS::L_BFGS(const size_t numBasis,
                      const size_t maxIterations,
                      const double armijoConstant,
                      const double wolfe,
                      const double minGradientNorm,
                      const double factr,
                      const size_t maxLineSearchTrials,
                      const double minStep,
                      const double maxStep) :
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
  // Nothing to do.
}

/**
 * Calculate the scaling factor, gamma, which is used to scale the Hessian
 * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal
 * (1989).
 *
 * @return The calculated scaling factor.
 * @param gradient The gradient at the initial point.
 * @param s Differences between the iterate and old iterate matrix.
 * @param y Differences between the gradient and the old gradient matrix.
 */
inline double L_BFGS::ChooseScalingFactor(const size_t iterationNum,
                                          const arma::mat& gradient,
                                          const arma::cube& s,
                                          const arma::cube& y)
{
  double scalingFactor = 1.0;
  if (iterationNum > 0)
  {
    int previousPos = (iterationNum - 1) % numBasis;
    // Get s and y matrices once instead of multiple times.
    const arma::mat& sMat = s.slice(previousPos);
    const arma::mat& yMat = y.slice(previousPos);
    scalingFactor = dot(sMat, yMat) / dot(yMat, yMat);
  }
  else
  {
    scalingFactor = 1.0 / sqrt(dot(gradient, gradient));
  }

  return scalingFactor;
}

/**
 * Find the L_BFGS search direction.
 *
 * @param gradient The gradient at the current point.
 * @param iterationNum The iteration number.
 * @param scalingFactor Scaling factor to use (see ChooseScalingFactor_()).
 * @param s Differences between the iterate and old iterate matrix.
 * @param y Differences between the gradient and the old gradient matrix.
 * @param searchDirection Vector to store search direction in.
 */
inline void L_BFGS::SearchDirection(const arma::mat& gradient,
                                    const size_t iterationNum,
                                    const double scalingFactor,
                                    const arma::cube& s,
                                    const arma::cube& y,
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
 * @param iterationNum Iteration number.
 * @param iterate Current point.
 * @param oldIterate Point at last iteration.
 * @param gradient Gradient at current point (iterate).
 * @param oldGradient Gradient at last iteration point (oldIterate).
 * @param s Differences between the iterate and old iterate matrix.
 * @param y Differences between the gradient and the old gradient matrix.
 */
inline void L_BFGS::UpdateBasisSet(const size_t iterationNum,
                                   const arma::mat& iterate,
                                   const arma::mat& oldIterate,
                                   const arma::mat& gradient,
                                   const arma::mat& oldGradient,
                                   arma::cube& s,
                                   arma::cube& y)
{
  // Overwrite a certain position instead of pushing everything in the vector
  // back one position.
  int overwritePos = iterationNum % numBasis;
  s.slice(overwritePos) = iterate - oldIterate;
  y.slice(overwritePos) = gradient - oldGradient;
}

/**
 * Perform a back-tracking line search along the search direction to calculate a
 * step size satisfying the Wolfe conditions.
 *
 * @param function Function to optimize.
 * @param functionValue Value of the function at the initial point.
 * @param iterate The initial point to begin the line search from.
 * @param gradient The gradient at the initial point.
 * @param searchDirection A vector specifying the search direction.
 * @param stepSize Variable the calculated step size will be stored in.
 *
 * @return false if no step size is suitable, true otherwise.
 */
template<typename FunctionType>
bool L_BFGS::LineSearch(FunctionType& function,
                        double& functionValue,
                        arma::mat& iterate,
                        arma::mat& gradient,
                        arma::mat& newIterateTmp,
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
    Warn << "L-BFGS line search direction is not a descent direction "
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
  double bestStepSize = 1.0;
  double bestObjective = std::numeric_limits<double>::max();

  while (true)
  {
    // Perform a step and evaluate the gradient and the function values at that
    // point.
    newIterateTmp = iterate;
    newIterateTmp += stepSize * searchDirection;
    functionValue = function.EvaluateWithGradient(newIterateTmp, gradient);
    if (functionValue < bestObjective)
    {
      bestStepSize = stepSize;
      bestObjective = functionValue;
    }
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
  iterate += bestStepSize * searchDirection;
  return true;
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
double L_BFGS::Optimize(FunctionType& function, arma::mat& iterate)
{
  // Use the Function<> wrapper to ensure the function has all of the functions
  // that we need.
  typedef Function<FunctionType> FullFunctionType;
  FullFunctionType& f = static_cast<FullFunctionType&>(function);

  // Check that we have all the functions we will need.
  traits::CheckFunctionTypeAPI<FullFunctionType>();

  // Ensure that the cubes holding past iterations' information are the right
  // size.  Also set the current best point value to the maximum.
  const size_t rows = iterate.n_rows;
  const size_t cols = iterate.n_cols;

  arma::mat newIterateTmp(rows, cols);
  arma::cube s(rows, cols, numBasis);
  arma::cube y(rows, cols, numBasis);

  // The old iterate to be saved.
  arma::mat oldIterate;
  oldIterate.zeros(iterate.n_rows, iterate.n_cols);

  // Whether to optimize until convergence.
  bool optimizeUntilConvergence = (maxIterations == 0);

  // The gradient: the current and the old.
  arma::mat gradient(iterate.n_rows, iterate.n_cols, arma::fill::zeros);
  arma::mat oldGradient(iterate.n_rows, iterate.n_cols, arma::fill::zeros);

  // The search direction.
  arma::mat searchDirection(iterate.n_rows, iterate.n_cols, arma::fill::zeros);

  // The initial function value and gradient.
  double functionValue = f.EvaluateWithGradient(iterate, gradient);
  double prevFunctionValue = functionValue;

  // The main optimization loop.
  for (size_t itNum = 0; optimizeUntilConvergence || (itNum != maxIterations);
       ++itNum)
  {
    prevFunctionValue = functionValue;

    // Break when the norm of the gradient becomes too small.
    //
    // But don't do this on the first iteration to ensure we always take at
    // least one descent step.
    if (itNum > 0 && (arma::norm(gradient, 2) < minGradientNorm))
    {
      Warn << "L-BFGS gradient norm too small (terminating successfully)."
          << std::endl;
      break;
    }

    // Break if the objective is not a number.
    if (std::isnan(functionValue))
    {
      Warn << "L-BFGS terminated with objective " << functionValue << "; "
          << "are the objective and gradient functions implemented correctly?"
          << std::endl;
      break;
    }

    // Choose the scaling factor.
    double scalingFactor = ChooseScalingFactor(itNum, gradient, s, y);

    // Build an approximation to the Hessian and choose the search
    // direction for the current iteration.
    SearchDirection(gradient, itNum, scalingFactor, s, y, searchDirection);

    // Save the old iterate and the gradient before stepping.
    oldIterate = iterate;
    oldGradient = gradient;

    if (!LineSearch(f, functionValue, iterate, gradient, newIterateTmp,
        searchDirection))
    {
      Warn << "Line search failed.  Stopping optimization." << std::endl;
      break; // The line search failed; nothing else to try.
    }

    // It is possible that the difference between the two coordinates is zero.
    // In this case we terminate successfully.
    if (accu(iterate != oldIterate) == 0)
    {
      Info << "L-BFGS step size of 0 (terminating successfully)."
          << std::endl;
      break;
    }

    // If we can't make progress on the gradient, then we'll also accept
    // a stable function value.
    const double denom = std::max(
        std::max(fabs(prevFunctionValue), fabs(functionValue)), 1.0);
    if ((prevFunctionValue - functionValue) / denom <= factr)
    {
      Info << "L-BFGS function value stable (terminating successfully)."
          << std::endl;
      break;
    }

    // Overwrite an old basis set.
    UpdateBasisSet(itNum, iterate, oldIterate, gradient, oldGradient, s, y);
  } // End of the optimization loop.

  return functionValue;
}

} // namespace ens

#endif // ENSMALLEN_LBFGS_LBFGS_IMPL_HPP

