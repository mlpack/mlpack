/**
 * @file lbfgs.cpp
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
#include "lbfgs.hpp"

namespace mlpack {
namespace optimization {

/**
 * Initialize the L_BFGS object.
 *
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
L_BFGS::L_BFGS(const size_t numBasis,
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
 * Calculate the scaling factor gamma which is used to scale the Hessian
 * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal (1989).
 *
 * @return The calculated scaling factor
 */
double L_BFGS::ChooseScalingFactor(const size_t iterationNum,
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
 * @param gradient The gradient at the current point
 * @param iterationNum The iteration number
 * @param scalingFactor Scaling factor to use (see ChooseScalingFactor_())
 * @param searchDirection Vector to store search direction in
 */
void L_BFGS::SearchDirection(const arma::mat& gradient,
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
 * @param iterationNum Iteration number
 * @param iterate Current point
 * @param oldIterate Point at last iteration
 * @param gradient Gradient at current point (iterate)
 * @param oldGradient Gradient at last iteration point (oldIterate)
 */
void L_BFGS::UpdateBasisSet(const size_t iterationNum,
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

} // namespace optimization
} // namespace mlpack
