/**
 * @file svrg_update.hpp
 * @author Marcus Edel
 *
 * Vanilla update for SARAH.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SARAH_SARAH_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SARAH_SARAH_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Vanilla update policy for SARAH.
 */
class SARAHUpdate
{
 public:
  /**
   * Update step for SARAH. The function parameters are updated in the negative
   * direction of the gradient.
   *
   * @param iterate Parameters that minimize the function.
   * @param v Unbiased estimator of the gradient.
   * @param gradient The current gradient matrix at time t.
   * @param gradient0 The old gradient matrix at time t - 1.
   * @param batchSize Batch size to be used for the given iteration.
   * @param stepSize Step size to be used for the given iteration.
   * @param vNorm The norm of the full gradient.
   */
  bool Update(arma::mat& iterate,
              arma::mat& v,
              const arma::mat& gradient,
              const arma::mat& gradient0,
              const size_t batchSize,
              const double stepSize,
              const double /* vNorm */)
  {
    v += (gradient - gradient0) / (double) batchSize;
    iterate -= stepSize * v;
    return false;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
