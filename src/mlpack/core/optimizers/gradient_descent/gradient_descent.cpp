/**
 * @file gradient_descent_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Simple gradient descent implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "gradient_descent.hpp"

namespace mlpack {
namespace optimization {

GradientDescent::GradientDescent(
    const double stepSize,
    const size_t maxIterations,
    const double tolerance) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack
