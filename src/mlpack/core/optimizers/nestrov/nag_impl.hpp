/**
 * @file nag_impl.hpp
 * @author Ryan Curtin
 * @author Kris Singh
 *
 * Implementation of stochastic gradient descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_NAG_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_NAG__IMPL_HPP

#include <mlpack/methods/regularized_svd/regularized_svd_function.hpp>

// In case it hasn't been included yet.
#include "nag.hpp"
#include "nestrov_update.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
NAG<DecomposableFunctionType>::NAG(DecomposableFunctionType& function,
                                   const double momentum,
                                   const double stepSize,
                                   const size_t maxIterations,
                                   const double tolerance,
                                   const bool shuffle):
nestrovupdate(NestrovUpdate(momentum)),
optimizer(SGD<DecomposableFunctionType, NestrovUpdate>(function, 
      stepSize, maxIterations, tolerance, shuffle, nestrovupdate))
{
}

} // namespace optimization
} // namespace mlpack

#endif
