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

// In case it hasn't been included yet.
#include "parallel_sgd.hpp"

namespace mlpack {
namespace optimization {

template <typename DecayPolicyType>
ParallelSGD<DecayPolicyType>::ParallelSGD(
    const size_t maxIterations,
    const size_t batchSize,
    const double tolerance,
    const DecayPolicyType& decayPolicy) :
    maxIterations(maxIterations),
    batchSize(batchSize),
    tolerance(tolerance),
    decayPolicy(decayPolicy)
{ /* Nothing to do. */ }

template <typename DecayPolicyType>
template <typename SparseFunctionType>
double ParallelSGD<DecayPolicyType>::Optimize(
    SparseFunctionType& function,
    arma::mat& iterate)
{
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  for (size_t i = 1; i <= maxIterations; ++i){
    overallObjective = 0;

    // Get the stepsize for this iteration
    double stepSize = decayPolicy.StepSize(i);

    arma::Col<size_t> visitationOrder;
    GenerateVisitationOrder(visitationOrder, function.NumFunctions());

    #pragma omp parallel
    {
      // Each processor gets a subset of the instances.
      // Each subset is of size batchSize.
      arma::Col<size_t> instances = ThreadShare(omp_get_thread_num(),
          visitationOrder);
      for (size_t j = 0; j < instances.n_elem; ++j)
      {
        // Each instance affects only some components of the decision variable.
        // So the gradient is sparse.
        arma::sp_mat gradient;

        // Evaluate the sparse gradient.
        function.Gradient(iterate, instances[j], gradient);

        // Update the decision variable with non-zero components of the
        // gradient.
        for (size_t i = 0; i < gradient.n_cols; ++i)
        {
          // Iterate over the non-zero elements.
          for (auto cur = gradient.begin_col(i); cur != gradient.end_col(i);
              ++cur)
          {
            #pragma omp atomic
            iterate(cur.row(), i) -= stepSize * gradient(cur.row(), i);
          }
        }
      }
    }

    // Evaluate the function
    overallObjective = 0;
    for (size_t j = 0; j < function.NumFunctions(); ++j)
    {
      overallObjective += function.Evaluate(iterate, j);
    }

    Log::Info << "\nObjective : " << overallObjective << " Iteration : " << i;
    if (std::abs(overallObjective - lastObjective) < tolerance)
    {
      Log::Info << "\nParallel SGD terminated with objective delta "
        << " within tolerance : " << overallObjective << std::endl;
      return overallObjective;
    }
    lastObjective = overallObjective;
  }
  Log::Info << "\n Parallel SGD terminated with objective : "
    << overallObjective << std::endl;
  return overallObjective;
}

template <typename DecayPolicyType>
void ParallelSGD<DecayPolicyType>::GenerateVisitationOrder(
        arma::Col<size_t>& visitationOrder, size_t numFunctions)
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t>>(0,
      (numFunctions - 1), numFunctions));
}

template <typename DecayPolicyType>
arma::Col<size_t> ParallelSGD<DecayPolicyType>::ThreadShare(
    size_t thread_id, const arma::Col<size_t>& visitationOrder)
{
  if (thread_id * batchSize >= visitationOrder.n_elem)
  {
    // No data for this thread.
    return arma::Col<size_t>();
  }
  else if ((thread_id + 1) * batchSize >= visitationOrder.n_elem)
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

} // namespace optimization
} // namespace mlpack

#endif
