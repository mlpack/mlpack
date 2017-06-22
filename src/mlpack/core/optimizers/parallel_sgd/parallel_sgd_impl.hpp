/**
 * @file parallel_sgd_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of Parallel Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "stepsize_policies/constant_step.hpp"

// In case it hasn't been included yet.
#include "parallel_sgd.hpp"

namespace mlpack {
namespace optimization {

template <typename SparseFunctionType, typename StepsizePolicyType>
ParallelSGD<SparseFunctionType, StepsizePolicyType>::ParallelSGD(
    SparseFunctionType& function, 
    const size_t maxIterations, 
    const double tolerance,
    const StepsizePolicyType stepPolicy) :
    function(function),
    maxIterations(maxIterations),
    tolerance(tolerance),
    stepPolicy(stepPolicy)
{ /* Nothing to do. */ }

template <typename SparseFunctionType, typename StepsizePolicyType>
double ParallelSGD<SparseFunctionType, StepsizePolicyType>::Optimize(
    SparseFunctionType& function, 
    arma::mat& iterate)
{
}

}
}

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
