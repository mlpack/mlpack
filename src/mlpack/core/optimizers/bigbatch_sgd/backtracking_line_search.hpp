/**
 * @file backtracking_line_search.hpp
 * @author Marcus Edel
 *
 * Definition of the backtracking line search technique as described in:
 * "Big Batch SGD: Automated Inference using Adaptive Batch Sizes" by
 * S. De et al.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BACKTRACKING_LINE_SEARCH_HPP
#define MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BACKTRACKING_LINE_SEARCH_HPP

namespace mlpack {
namespace optimization {

/**
 * Definition of the backtracking line search algorithm based on the
 * Armijo–Goldstein condition to determine the maximum amount to move along the
 * given search direction.
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
class BacktrackingLineSearch
{
 public:
  /**
   * Construct the BacktrackingLineSearch object with the given function and
   * parameters. The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task at
   * hand.
   *
   * @param function Function to be optimized (minimized).
   */
  BacktrackingLineSearch(const double searchParameter = 0.1) :
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
              const double /* sampleVariance */,
              const size_t offset,
              const size_t /* batchSize */,
              const size_t backtrackingBatchSize,
              const bool reset)
  {
    if (reset)
      stepSize *= 2;

    double overallObjective = function.Evaluate(iterate, offset,
        backtrackingBatchSize);

    arma::mat iterateUpdate = iterate - (stepSize * gradient);
    double overallObjectiveUpdate = function.Evaluate(iterateUpdate,
        offset, backtrackingBatchSize);

    while (overallObjectiveUpdate >
        (overallObjective + searchParameter * stepSize * gradientNorm))
    {
      stepSize /= 2;

      iterateUpdate = iterate - (stepSize * gradient);
      overallObjectiveUpdate = function.Evaluate(iterateUpdate,
        offset, backtrackingBatchSize);
    }
  }

 private:
  //! The search parameter for each iteration.
  double searchParameter;
};

} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BACKTRACKING_LINE_SEARCH_HPP
