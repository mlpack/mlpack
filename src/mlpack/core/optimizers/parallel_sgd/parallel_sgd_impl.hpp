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

#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace optimization {

template <typename DecayPolicyType>
ParallelSGD<DecayPolicyType>::ParallelSGD(
    const size_t maxIterations,
    const size_t threadShareSize,
    const double tolerance,
    const bool shuffle,
    const DecayPolicyType& decayPolicy) :
    maxIterations(maxIterations),
    threadShareSize(threadShareSize),
    tolerance(tolerance),
    shuffle(shuffle),
    decayPolicy(decayPolicy)
{ /* Nothing to do. */ }

template <typename DecayPolicyType>
template <typename SparseFunctionType>
double ParallelSGD<DecayPolicyType>::Optimize(
    SparseFunctionType& function,
    arma::mat& iterate)
{
  // Check that we have all the functions that we need.
  traits::CheckSparseFunctionTypeAPI<SparseFunctionType>();

  double overallObjective = DBL_MAX;
  double lastObjective;

  // The order in which the functions will be visited.
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (function.NumFunctions() - 1), function.NumFunctions());

  // Iterate till the objective is within tolerance or the maximum number of
  // allowed iterations is reached. If maxIterations is 0, this will iterate
  // till convergence.
  for (size_t i = 1; i != maxIterations; ++i)
  {
    // Calculate the overall objective.
    lastObjective = overallObjective;

    overallObjective = function.Evaluate(iterate);

    // Output current objective function.
    Log::Info << "Parallel SGD: iteration " << i << ", objective "
      << overallObjective << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Log::Warn << "Parallel SGD: converged to " << overallObjective
        << "; terminating with failure. Try a smaller step size?"
        << std::endl;
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Log::Info << "SGD: minimized within tolerance " << tolerance << "; "
        << "terminating optimization." << std::endl;
      return overallObjective;
    }

    // Get the stepsize for this iteration
    double stepSize = decayPolicy.StepSize(i);

    // Shuffle for uniform sampling of functions by each thread.
    if (shuffle)
    {
      // Determine order of visitation.
      std::shuffle(visitationOrder.begin(), visitationOrder.end(),
          mlpack::math::randGen);
    }

    #pragma omp parallel
    {
      // Each processor gets a subset of the instances.
      // Each subset is of size threadShareSize
      size_t threadId = 0;
      #ifdef HAS_OPENMP
        threadId = omp_get_thread_num();
      #endif

      for (size_t j = threadId * threadShareSize;
          j < (threadId + 1) * threadShareSize && j < visitationOrder.n_elem;
          ++j)
      {
        // Each instance affects only some components of the decision variable.
        // So the gradient is sparse.
        arma::sp_mat gradient;

        // Evaluate the sparse gradient.
        function.Gradient(iterate, visitationOrder[j], gradient, 1);

        // Update the decision variable with non-zero components of the
        // gradient.
        for (size_t i = 0; i < gradient.n_cols; ++i)
        {
          // Iterate over the non-zero elements.
          for (arma::sp_mat::iterator cur = gradient.begin_col(i);
              cur != gradient.end_col(i); ++cur)
          {
            #pragma omp atomic
            iterate(cur.row(), i) -= stepSize * (*cur);
          }
        }
      }
    }
  }

  Log::Info << "\n Parallel SGD terminated with objective : "
    << overallObjective << std::endl;
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
