/**
 * @file sag_update.hpp
 * @authors Prabhat Sharma and Marcus Edel
 *
 * Vanilla update for SAG, a stochastic gradient method for
 * optimizing the sum of a finite set of smooth functions, where
 * the sum is strongly convex.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SAG_SAG_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SAG_SAG_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
* Vanilla update policy for SAG, a stochastic gradient method for
* optimizing the sum of a finite set of smooth functions, where
* the sum is strongly convex.
* For more information, please refer to:
*
* @code
* @article{schmidt2017minimizing,
* title     = {Minimizing finite sums with the stochastic average gradient},
* author    = {Schmidt, Mark and Le Roux, Nicolas and Bach, Francis},
* journal   = {Mathematical Programming},
* year      = {2017},
* publisher = {Springer}
* }
* @endcode
*
* The following update scheme is used to update SAG in every iteration:
*/
class SAGUpdate
{
 public:
  /**
   * The Initialize method is called by SAG Optimizer method before the start
   * of the iteration update process. The vanilla update doesn't initialize
   * anything.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t /* rows */, const size_t /* cols */)
  { /* Do nothing. */ }

  /**
   * Update step for SAG. The function parameters are updated in the negative
   * direction of the gradient.
   *
   * @param iterate Parameters that minimize the function.
   * @param avgGradient The computed average gradient of all functions.
   * @param gradient The current gradient matrix at time t.
   * @param gradient0 The old gradient matrix at time t - 1.
   * @param stepSize Step size to be used for the given iteration.
   * @param batches Total number of batches present.
   */
  void Update(arma::mat& iterate,
              const arma::mat& avgGradient,
              const arma::mat& gradient,
              const arma::mat& gradient0,
              const double stepSize,
              const size_t batches)
  {
    // Perform the vanilla SAG update.
    iterate -= stepSize * (avgGradient + (gradient - gradient0) / batches);
  }
};

} // namespace optimization
} // namespace mlpack

#endif

