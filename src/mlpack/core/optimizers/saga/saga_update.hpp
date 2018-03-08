/**
 * @file saga_update.hpp
 * @author Prabhat Sharma
 *
 * Vanilla update for stochastic variance reduced gradient (SVRG).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SAGA_SAGA_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SAGA_SAGA_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
  namespace optimization {

/**
 * Vanilla update policy for Stochastic variance reduced gradient (SVRG).
 * The following update scheme is used to update SGD in every iteration:
 */
    class SAGAUpdate
    {
    public:
      /**
       * The Initialize method is called by SVRG Optimizer method before the start
       * of the iteration update process. The vanilla update doesn't initialize
       * anything.
       *
       * @param rows Number of rows in the gradient matrix.
       * @param cols Number of columns in the gradient matrix.
       */
      void Initialize(const size_t /* rows */, const size_t /* cols */)
      { /* Do nothing. */ }

      /**
       * Update step for SVRG. The function parameters are updated in the negative
       * direction of the gradient.
       *
       * @param iterate Parameters that minimize the function.
       * @param fullGradient The computed full gradient.
       * @param gradient The current gradient matrix at time t.
       * @param gradient0 The old gradient matrix at time t - 1.
       * @param batchSize Batch size to be used for the given iteration.
       * @param stepSize Step size to be used for the given iteration.
       */
      void Update(arma::mat& iterate,
                  const arma::mat& avgGradient,
                  const arma::mat& gradient,
                  const arma::mat& gradient0,
                  const double stepSize)
      {
        // Perform the vanilla SVRG update.
        iterate -= stepSize * (avgGradient + (gradient - gradient0));
      }
    };

  } // namespace optimization
} // namespace mlpack

#endif

