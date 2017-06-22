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
  for (size_t i = 0; i < maxIterations; ++i){
    double stepSize = stepPolicy.StepSize(i);
    function.Initialize();
    #pragma omp parallel
    {
      // Each processor gets a subset of the instances
      arma::Col<size_t> instances =
        function.RandomInstanceSet(omp_get_thread_num(), omp_get_num_threads());
      for (size_t i = 0; i < instances.n_elem; ++i){
        // Each instance affects only some components of the decision variable
        arma::Col<size_t> components = function.Components(instances[i]);
        // Evaluate the gradient
        // FIXME: Should evaluate only at the components required
        arma::vec gradient = function.Gradient(iterate, instances[i]);

        for(size_t j = 0; j < components.n_elem; ++i){
          #pragma omp atomic
          iterate[components[j]] -= stepSize * gradient[components[j]];
        }
      }
    }
  }
}

}
}

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
