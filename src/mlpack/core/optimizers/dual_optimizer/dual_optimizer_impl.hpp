/**
 * @file dual_optimizer_impl.hpp
 * @author Shikhar Jaiswal
 *
 * Implementation of the Dual Optimizer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_DUAL_OPTIMIZER_DUAL_OPTIMIZER_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_DUAL_OPTIMIZER_DUAL_OPTIMIZER_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_optimizer.hpp"

namespace mlpack {
namespace optimization {

template<class DiscriminatorOptimizer, class GeneratorOptimizer>
DualOptimizer<DiscriminatorOptimizer, GeneratorOptimizer>::DualOptimizer(
    DiscriminatorOptimizer& optDiscriminator,
    GeneratorOptimizer& optGenerator):
    optDiscriminator(optDiscriminator),
    optGenerator(optGenerator)
{ /* Nothing to do. */ }

} // namespace optimization
} // namespace mlpack

#endif
