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
void ParallelSGD<
    SparseFunctionType,
    StepsizePolicyType>::GenerateVisitationOrder()
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (function.numFunctions() - 1), function.NumFunctions()));
}

template <typename SparseFunctionType, typename StepsizePolicyType>
arma::Col<size_t> ParallelSGD<
    SparseFunctionType,
    StepsizePolicyType>::ThreadShare(size_t thread_id, size_t max_threads)
{
  arma::Col<size_t> threadShare;
  size_t examplesPerThread = std::floor(function.NumFunctions() / max_threads);
  if (thread_id == max_threads - 1){
    // The last thread gets the remaining instances
    threadShare = visitationOrder.subvec(thread_id * examplesPerThread,
        function.numFunctions() - 1);
  }
  else 
  {
    // An equal distribution of data
    threadShare = visitationOrder.subvec(thread_id * examplesPerThread,
        (thread_id + 1) * examplesPerThread - 1);
  }
  return threadShare;
}

template <typename SparseFunctionType, typename StepsizePolicyType>
double ParallelSGD<SparseFunctionType, StepsizePolicyType>::Optimize(
    SparseFunctionType& function, 
    arma::mat& iterate)
{
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  for (size_t i = 1; i != maxIterations; ++i){
    overallObjective = 0;
    double stepSize = stepPolicy.StepSize(i);
    GenerateVisitationOrder();
    #pragma omp parallel
    {
      // Each processor gets a subset of the instances
      arma::Col<size_t> instances = ThreadShare(omp_get_thread_num(),
            omp_get_num_threads());
      for (size_t j = 0; j < instances.n_elem; ++j)
      {
        // Each instance affects only some components of the decision variable
        arma::Col<size_t> components = function.Components(instances[j]);
        // Evaluate the gradient
        arma::vec gradient;
        function.Gradient(iterate, instances[j], gradient);

        for(size_t k = 0; k < components.n_elem; ++k)
        {
          #pragma omp atomic
          iterate[components[k]] -= stepSize * gradient[components[k]];
        }
      }
    }
    // Evaluate the function
    overallObjective = 0;
    for(size_t j = 0; j < function.NumFunctions(); ++j){
      overallObjective += function.Evaluate(iterate, j);
    }
    if(std::abs(overallObjective - lastObjective) < tolerance){
      return overallObjective;
    }
    lastObjective = overallObjective;
  }
  return overallObjective;
}

}
}

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
