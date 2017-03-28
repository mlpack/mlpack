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
#ifndef MLPACK_CORE_OPTIMIZERS_ADA_DELTA_ADA_DELTA_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_ADA_DELTA_ADA_DELTA_IMPL_HPP

#include "ada_delta.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
AdaDelta<DecomposableFunctionType>::AdaDelta(DecomposableFunctionType& function,
                                             const double stepSize,
                                             const double rho,
                                             const double epsilon,
                                             const size_t maxIterations,
                                             const double tolerance,
                                             const bool shuffle) :
    optimizer(function,
              stepSize,
              maxIterations,
              tolerance,
              shuffle,
              AdaDeltaUpdate(rho, epsilon))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif
