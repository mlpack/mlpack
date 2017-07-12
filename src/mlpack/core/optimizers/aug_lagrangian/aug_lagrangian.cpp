/**
 * @file aug_lagrangian_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangian class (Augmented Lagrangian optimization
 * method).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include "aug_lagrangian.hpp"

namespace mlpack {
namespace optimization {

AugLagrangian::AugLagrangian() :
    lbfgsInternal(),
    lbfgs(lbfgsInternal)
{
  lbfgs.MaxIterations() = 1000;
}

} // namespace optimization
} // namespace mlpack

