/**
 * @file test_function.hpp
 * @author Adeel Ahmad
 *
 * Simple test function for Particle Swarm Optimization.
 * The equation below corresponds to the Rosenbrock function,
 * which is non-convex and used as a performance test problem
 * for optimization algorithms.
 *
 * \f[
 * f(\mathbf{x}) = \sum_{i=1}^{N-1} 100 (x_{i+1} - x_i^2 )^2 +
 * (1-x_i)^2 \quad \mbox{where} \quad \mathbf{x} =
 * [x_1, \ldots, x_N] \in \mathbb{R}^N
 * \f]
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_TEST_FUNC_FW_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_TEST_FUNC_FW_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Simple test function for Particle Swarm Optimization:
 *
 * \f[
 * f(\mathbf{x}) = \sum_{i=1}^{N-1} 100 (x_{i+1} - x_i^2 )^2 +
 * (1-x_i)^2 \quad \mbox{where} \quad \mathbf{x} =
 * [x_1, \ldots, x_N] \in \mathbb{R}^N
 * \f]
 *
 */
class PSOTestFunction
{
 public:
  PSOTestFunction()
  { /* Nothing to do. */ }

  /**
   * Evaluation of the function.
   *
   * @param coords Input vector x.
   * @param dimension Dimension of the search space.
   */
  double Evaluate(const arma::mat& coords, size_t dimension = 2)
  {
    double sum = 0;
    for (size_t i = 0; i < dimension; ++i)
    {
      sum += 100. * std::pow((coords[i + 1] - std::pow(coords[i], 2)), 2) +
        std::pow((1 - coords[i]), 2);
    }
    return sum;
  }
};

}  // namespace optimization
}  // namespace mlpack

#endif
