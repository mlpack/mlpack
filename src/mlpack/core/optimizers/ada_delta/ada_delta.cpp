/**
 * @file ada_delta_impl.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 *
 * Implementation of the AdaDelta optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "ada_delta.hpp"

namespace mlpack {
namespace optimization {

AdaDelta::AdaDelta(const double stepSize,
                   const size_t batchSize,
                   const double rho,
                   const double epsilon,
                   const size_t maxIterations,
                   const double tolerance,
                   const bool shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              AdaDeltaUpdate(rho, epsilon))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack
