/**
 * @file adaptive_stepsize.hpp
 * @author Marcus Edel
 *
 * Definition of the adaptive stepsize technique as described in:
 * "Big Batch SGD: Automated Inference using Adaptive Batch Sizes" by
 * S. De et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_ADAPTIVE_STEPSIZE_HPP
#define MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_ADAPTIVE_STEPSIZE_HPP

namespace mlpack {
namespace optimization {

/**
 * Definition of the adaptive stepize technique, a non-monotonic stepsize scheme
 * that uses curvature estimates to propose new stepsize choices.
 * direction.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{De2017,
 *   title   = {Big Batch {SGD:} Automated Inference using Adaptive Batch
 *              Sizes},
 *   author  = {Soham De and Abhay Kumar Yadav and David W. Jacobs and
                Tom Goldstein},
 *   journal = {CoRR},
 *   year    = {2017},
 *   url     = {http://arxiv.org/abs/1610.05792},
 * }
 * @endcode
 */
class AdaptiveStepsize
{
 public:
  /**
   * Construct the AdaptiveStepsize object with the given function and
   * parameters. The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task at
   * hand.
   *
   * @param backtrackStepSize The backtracking step size for each iteration.
   * @param searchParameter The backtracking search parameter for each
   *        iteration.
   */
  AdaptiveStepsize(const double backtrackStepSize = 0.1,
                   const double searchParameter = 0.1) :
      backtrackStepSize(backtrackStepSize),
      searchParameter(searchParameter)
  { /* Nothing to do here. */ }

  /**
   * This function is called in each iteration.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to be optimized (minimized).
   * @param stepSize Step size to be used for the given iteration.
   * @param iterate Parameters that minimize the function.
   * @param gradient The gradient matrix.
   * @param gradientNorm The gradient norm to be used for the given iteration.
   * @param offset The batch offset to be used for the given iteration.
   * @param batchSize Batch size to be used for the given iteration.
   * @param backtrackingBatchSize Backtracking batch size to be used for the
   *        given iteration.
   * @param reset Reset the step size decay parameter.
   */
  template<typename DecomposableFunctionType>
  void Update(DecomposableFunctionType& function,
              double& stepSize,
              arma::mat& iterate,
              const arma::mat& gradient,
              const double gradientNorm,
              const double sampleVariance,
              const size_t offset,
              const size_t batchSize,
              const size_t backtrackingBatchSize,
              const bool /* reset */)
  {
    Backtracking(function, stepSize, iterate, gradient, gradientNorm, offset,
        backtrackingBatchSize);

    // Update the iterate.
    iterate -= stepSize * gradient;

    double stepSizeDecay = 0;
    if (batchSize < function.NumFunctions())
    {
      stepSizeDecay = (1 - (1 / ((double) batchSize - 1) * sampleVariance) /
          (batchSize * gradientNorm)) / batchSize;
    }
    else
    {
      stepSizeDecay = 1 / function.NumFunctions();
    }

    // Stepsize smoothing.
    stepSize *= (1 - ((double) batchSize / function.NumFunctions()));
    stepSize += stepSizeDecay * ((double) batchSize / function.NumFunctions());

    Backtracking(function, stepSize, iterate, gradient, gradientNorm, offset,
        backtrackingBatchSize);
  }

  //! Get the backtracking step size.
  double BacktrackStepSize() const { return backtrackStepSize; }
  //! Modify the backtracking step size.
  double& BacktrackStepSize() { return backtrackStepSize; }

  //! Get the search parameter.
  double SearchParameter() const { return searchParameter; }
  //! Modify the search parameter.
  double& SearchParameter() { return searchParameter; }

 private:
  /**
   * Definition of the backtracking line search algorithm based on the
   * Armijo–Goldstein condition to determine the maximum amount to move along
   * the given search direction.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @param function Function to be optimized (minimized).
   * @param stepSize Step size to be used for the given iteration.
   * @param iterate Parameters that minimize the function.
   * @param gradient The gradient matrix.
   * @param gradientNorm The gradient norm to be used for the given iteration.
   * @param offset The batch offset to be used for the given iteration.
   * @param backtrackingBatchSize The backtracking batch size.
   */
  template<typename DecomposableFunctionType>
  void Backtracking(DecomposableFunctionType& function,
                    double& stepSize,
                    const arma::mat& iterate,
                    const arma::mat& gradient,
                    const double gradientNorm,
                    const size_t offset,
                    const size_t backtrackingBatchSize)
  {
    double overallObjective = function.Evaluate(iterate, offset,
        backtrackingBatchSize);

    arma::mat iterateUpdate = iterate - (stepSize * gradient);
    double overallObjectiveUpdate = function.Evaluate(iterateUpdate, offset,
        backtrackingBatchSize);

    while (overallObjectiveUpdate >
        (overallObjective + searchParameter * stepSize * gradientNorm))
    {
      stepSize *= backtrackStepSize;

      iterateUpdate = iterate - (stepSize * gradient);
      overallObjectiveUpdate = function.Evaluate(iterateUpdate, offset,
          backtrackingBatchSize);
    }
  }

  //! The backtracking step size for each iteration.
  double backtrackStepSize;

  //! The search parameter for each iteration.
  double searchParameter;
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_ADAPTIVE_STEPSIZE_HPP
