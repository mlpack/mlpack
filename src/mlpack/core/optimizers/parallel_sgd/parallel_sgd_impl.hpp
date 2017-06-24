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

// In case it hasn't been included yet.
#include "parallel_sgd.hpp"

namespace mlpack {
namespace optimization {

template <typename SparseFunctionType, typename DecayPolicyType>
ParallelSGD<SparseFunctionType, DecayPolicyType>::ParallelSGD(
    SparseFunctionType& function, 
    const size_t maxIterations, 
    const size_t batchSize,
    const double tolerance,
    const DecayPolicyType& decayPolicy) :
    function(function),
    maxIterations(maxIterations),
    batchSize(batchSize),
    tolerance(tolerance),
    decayPolicy(decayPolicy)
{ /* Nothing to do. */ }



template <typename SparseFunctionType, typename StepsizePolicyType>
double ParallelSGD<SparseFunctionType, StepsizePolicyType>::Optimize(
    SparseFunctionType& function, 
    arma::mat& iterate)
{
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  for (size_t i = 1; i != maxIterations; ++i){
    overallObjective = 0;

    // Get the stepsize for this iteration
    double stepSize = decayPolicy.StepSize(i);
    arma::Col<size_t> visitationOrder;
    GenerateVisitationOrder(visitationOrder);

    #pragma omp parallel
    {
      // Each processor gets a subset of the instances
      // Each subset is of size batchSize
      arma::Col<size_t> instances = ThreadShare(omp_get_thread_num(),
          visitationOrder);
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
    for(size_t j = 0; j < function.NumFunctions(); ++j)
    {
      overallObjective += function.Evaluate(iterate, j);
    }

    if(std::abs(overallObjective - lastObjective) < tolerance)
    {
      return overallObjective;
    }
    lastObjective = overallObjective;
  }
  return overallObjective;
}

template <typename SparseFunctionType, typename StepsizePolicyType>
void ParallelSGD<
    SparseFunctionType,
    StepsizePolicyType>::GenerateVisitationOrder(
        arma::Col<size_t>& visitationOrder)
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
        (function.NumFunctions() - 1), function.NumFunctions()));
}

template <typename SparseFunctionType, typename StepsizePolicyType>
arma::Col<size_t> ParallelSGD<
    SparseFunctionType,
    StepsizePolicyType>::ThreadShare(size_t thread_id,
                                     const arma::Col<size_t>& visitationOrder)
{
  if(thread_id * batchSize >= visitationOrder.n_elem)
  {
    // No data for this thread.
    return arma::Col<size_t>();
  }
  else if((thread_id + 1) * batchSize >= visitationOrder.n_elem)
  {
    // The last few elements.
    return visitationOrder.subvec(thread_id * batchSize,
        visitationOrder.n_elem - 1);
  }
  else
  {
    // Equal distribution of batchSize examples to each thread.
    return visitationOrder.subvec(thread_id * batchSize,
        (thread_id + 1) * batchSize - 1);
  }
}

}
}

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
