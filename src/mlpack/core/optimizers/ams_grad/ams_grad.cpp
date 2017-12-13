/**
 * @file ams_grad_impl.hpp
 * @author Haritha Nair
 *
 * Implementation of the AMSGrad Optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
// In case it hasn't been included yet.
#include "ams_grad.hpp"

namespace mlpack {
namespace optimization {

AmsGrad::AmsGrad(
    const double stepSize,
    const size_t batchSize,
    const double beta1,
    const double beta2,
    const double epsilon,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              AmsGradUpdate(epsilon, beta1, beta2))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

