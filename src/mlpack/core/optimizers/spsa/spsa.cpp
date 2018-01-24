/**
 * @file spsa_update.hpp
 * @author N Rajiv Vaidyanathan
 *
 * SPSA update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "spsa.hpp"

namespace mlpack {
namespace optimization {

SPSA::SPSA(const double stepSize,
           const size_t batchSize,
           const float& alpha,
           const float& gamma,
           const float& a,
           const float& c,
           const size_t& maxIterations,
           const double& tolerance,
           const bool& shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              SPSAUpdate(batchSize, alpha, gamma,
                            a, c, maxIterations))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack
