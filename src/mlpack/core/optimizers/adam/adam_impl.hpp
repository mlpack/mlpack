/**
 * @file adam_impl.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Marcus Edel
 * @author Vivek Pal
 *
 * Implementation of the Adam, AdaMax, AMSGrad, Nadam and NadaMax optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_ADAM_ADAM_IMPL_HPP

// In case it hasn't been included yet.
#include "adam.hpp"

namespace mlpack {
namespace optimization {

template<typename UpdateRule>
AdamType<UpdateRule>::AdamType(
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
              UpdateRule(epsilon, beta1, beta2))
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif
