/**
 * @file random_selection.hpp
 * @author Marcus Edel
 *
 * Randomly select dataset points for use in the Evaluation step.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_RANDOM_SELECTION_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_RANDOM_SELECTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/*
 * Randomly select dataset points for use in the Evaluation step.
 */
class RandomSelection
{
 public:
  /**
   * Constructor for the random selection strategy.
   *
   * @param fraction The dataset fraction used for the selection (Default 0.3).
   */
  RandomSelection(const double fraction = 0.3) : fraction(fraction)
  {
    // Nothing to do here.
  }

  //! Get the dataset fraction.
  double Fraction() const { return fraction; }
  //! Modify the dataset fraction.
  double& Fraction() { return fraction; }

  /**
   * Randomly select dataset points to calculate the objective function.
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
    for (size_t f = 0; f < std::floor(numFunctions * fraction); f += batchSize)
    {
      const size_t selection = math::RandInt(0, numFunctions);
      const size_t effectiveBatchSize = std::min(batchSize,
          numFunctions - selection);

      objective += function.Evaluate(iterate, selection, effectiveBatchSize);
    }

    return objective;
  }

 private:
  //! Dataset fraction parameter.
  double fraction;
};

} // namespace optimization
} // namespace mlpack

#endif
