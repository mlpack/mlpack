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

AdaGrad::AdaGrad(const double stepSize = 0.01,
                 const size_t batchSize = 32,
                 const float& alpha = 0.602,
                 const float& gamma = 0.101,
                 const float& a = 1e-6,
                 const float& c = 0.01,
                 const size_t maxIterations = 100000,
                 const double tolerance = 1e-5,
                 const bool shuffle = true) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              AdaGradUpdate(batchSize, alpha, gamma,
                            a, c, maxIterations))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack
