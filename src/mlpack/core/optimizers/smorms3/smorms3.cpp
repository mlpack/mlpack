/**
 * @file smorms3_impl.hpp
 * @author Vivek Pal
 *
 * Implementation of the SMORMS3 constructor.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "smorms3.hpp"

namespace mlpack {
namespace optimization {

SMORMS3::SMORMS3(const double stepSize,
                 const size_t batchSize,
                 const double epsilon,
                 const size_t maxIterations,
                 const double tolerance,
                 const bool shuffle) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              SMORMS3Update(epsilon))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack
