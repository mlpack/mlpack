/**
 * @file inertia_weight.hpp
 *
 * PSO with inertia weight.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PSO_INERTIA_WEIGHT_HPP
#define MLPACK_CORE_OPTIMIZERS_PSO_INERTIA_WEIGHT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/*
 * PSO with inertia weight.
 */
class InertiaWeight
{
 public:
  /**
   * Update the particles' velocity and position.
   *
   * @tparam DecomposableFunctionType Type of the function to be evaluated.
   * @param function Function to optimize.
   * @param batchSize Batch size to use for each step.
   * @param iterate starting point.
   */
  template<typename DecomposableFunctionType>
  double UpdateParameters(DecomposableFunctionType& function,
                      const size_t batchSize,
                      const arma::mat& iterate)
  {
    // TODO: Implement!
  }
};

} // namespace optimization
} // namespace mlpack

#endif
