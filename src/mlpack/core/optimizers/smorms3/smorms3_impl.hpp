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
#ifndef MLPACK_CORE_OPTIMIZERS_SMORMS3_SMORMS3_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SMORMS3_SMORMS3_IMPL_HPP

// In case it hasn't been included yet.
#include "smorms3.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
SMORMS3<DecomposableFunctionType>::SMORMS3(DecomposableFunctionType& function,
                                           const double stepSize,
                                           const double epsilon,
                                           const size_t maxIterations,
                                           const double tolerance,
                                           const bool shuffle) :
    optimizer(function,
              stepSize,
              maxIterations,
              tolerance,
              shuffle,
              SMORMS3Update(epsilon))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif
