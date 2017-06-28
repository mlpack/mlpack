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
 * Evaluate the function at the given iterate point and store the result if
 * it is a new minimum.
 *
 * @return The value of the function
 */
template<typename FunctionType>
double L_BFGS::Evaluate(FunctionType& function,
                        const arma::mat& iterate,
                        std::pair<arma::mat, double>& minPointIterate)
{
  // Evaluate the function and keep track of the minimum function
  // value encountered during the optimization.
  const double functionValue = function.Evaluate(iterate);

  if (functionValue < minPointIterate.second)
  {
    minPointIterate.first = iterate;
    minPointIterate.second = functionValue;
  }

  return functionValue;
}

/**
 * Check to make sure that the norm of the gradient is not smaller than 1e-10.
 * Currently that value is not configurable.
 *
 * @return (norm < minGradientNorm)
 */
inline bool L_BFGS::GradientNormTooSmall(const arma::mat& gradient)
{
  const double norm = arma::norm(gradient, 2);

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
bool L_BFGS::LineSearch(FunctionType& function,
                        double& functionValue,
                        arma::mat& iterate,
                        arma::mat& gradient,
                        arma::mat& newIterateTmp,
                        std::pair<arma::mat, double>& minPointIterate,
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
    functionValue = Evaluate(function, newIterateTmp, minPointIterate);
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
  // Ensure that the cubes holding past iterations' information are the right
  // size.  Also set the current best point value to the maximum.
  const size_t rows = iterate.n_rows;
  const size_t cols = iterate.n_cols;

  arma::mat newIterateTmp(rows, cols);
  arma::cube s(rows, cols, numBasis);
  arma::cube y(rows, cols, numBasis);
  std::pair<arma::mat, double> minPointIterate;
  minPointIterate.second = std::numeric_limits<double>::max();

  // The old iterate to be saved.
  arma::mat oldIterate;
  oldIterate.zeros(iterate.n_rows, iterate.n_cols);

  // Whether to optimize until convergence.
  bool optimizeUntilConvergence = (maxIterations == 0);

  // The initial function value.
  double functionValue = Evaluate(function, iterate, minPointIterate);
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
        function.Evaluate(iterate) << ", gradient norm "
        << arma::norm(gradient, 2) << ", "
        << ((prevFunctionValue - functionValue) /
            std::max(std::max(fabs(prevFunctionValue),
                              fabs(functionValue)), 1.0))
        << "." << std::endl;

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
    double scalingFactor = ChooseScalingFactor(itNum, gradient, s, y);

    // Build an approximation to the Hessian and choose the search
    // direction for the current iteration.
    SearchDirection(gradient, itNum, scalingFactor, s, y, searchDirection);

    // Save the old iterate and the gradient before stepping.
    oldIterate = iterate;
    oldGradient = gradient;

    // Do a line search and take a step.
    if (!LineSearch(function, functionValue, iterate, gradient, newIterateTmp,
        minPointIterate, searchDirection))
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
    const double denom = std::max(
        std::max(fabs(prevFunctionValue), fabs(functionValue)), 1.0);
    if ((prevFunctionValue - functionValue) / denom <= factr)
    {
      Log::Debug << "L-BFGS function value stable (terminating successfully)."
          << std::endl;
      break;
    }

    // Overwrite an old basis set.
    UpdateBasisSet(itNum, iterate, oldIterate, gradient, oldGradient, s, y);
  } // End of the optimization loop.

  return function.Evaluate(iterate);
}

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP

