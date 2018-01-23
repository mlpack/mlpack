/**
 * @file full_selection.hpp
 * @author Marcus Edel
 *
 * Select the full dataset for use in the Evaluation step.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_FULL_SELECTION_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_FULL_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/*
 * Select the full dataset for use in the Evaluation step.
 */
class FullSelection
{
 public:
  /**
   * Select the full dataset to calculate the objective function.
   *
   * @tparam DecomposableFunctionType Type of the function to be evaluated.
   * @param function Function to optimize.
   * @param batchSize Batch size to use for each step.
   * @param iterate starting point.
   */
  template<typename DecomposableFunctionType>
  double Select(DecomposableFunctionType& function,
                      const size_t batchSize,
                      const arma::mat& iterate)
  {
    // Find the number of functions to use.
    const size_t numFunctions = function.NumFunctions();

    double objective = 0;
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      objective += function.Evaluate(iterate, f, effectiveBatchSize);
    }

    return objective;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
