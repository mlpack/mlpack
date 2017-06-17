/**
 * @file rmsprop_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * Implementation of the RMSProp constructor.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_RMSPROP_RMSPROP_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_RMSPROP_RMSPROP_IMPL_HPP

// In case it hasn't been included yet.
#include "rmsprop.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
RMSProp<DecomposableFunctionType>::RMSProp(DecomposableFunctionType& function,
                                           const double stepSize,
                                           const double alpha,
                                           const double epsilon,
                                           const size_t maxIterations,
                                           const double tolerance,
                                           const bool shuffle) :
    optimizer(function,
              stepSize,
              maxIterations,
              tolerance,
              shuffle,
              RMSPropUpdate(epsilon,
                            alpha))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif
