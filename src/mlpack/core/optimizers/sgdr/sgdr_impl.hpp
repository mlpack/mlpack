/**
 * @file sgdr_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of SGDR method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGDR_SGDR_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SGDR_SGDR_IMPL_HPP

// In case it hasn't been included yet.
#include "sgdr.hpp"

namespace mlpack {
namespace optimization {

template<
    typename DecomposableFunctionType,
    typename UpdatePolicyType,
    typename DecayPolicyType
>
template<typename PolicyType>
SGDR<
    DecomposableFunctionType,
    UpdatePolicyType,
    DecayPolicyType
>::SGDR(DecomposableFunctionType& function,
        const size_t epochRestart,
        const double multFactor,
        const size_t batchSize,
        const double stepSize,
        const size_t maxIterations,
        const double tolerance,
        const bool shuffle,
        const size_t snapshots,
        const UpdatePolicyType& updatePolicy,
        const typename std::enable_if_t<std::is_same<
            PolicyType, SnapshotEnsembles>::value>* /* junk */) :
    function(function),
    batchSize(batchSize),
    decayPolicy(SnapshotEnsembles(epochRestart,
                                  multFactor,
                                  stepSize,
                                  batchSize,
                                  function.NumFunctions(),
                                  maxIterations,
                                  snapshots)),
    optimizer(OptimizerType(function,
                            batchSize,
                            stepSize,
                            maxIterations,
                            tolerance,
                            shuffle,
                            updatePolicy,
                            decayPolicy))
{
  /* Nothing to do here */
}

template<
    typename DecomposableFunctionType,
    typename UpdatePolicyType,
    typename DecayPolicyType
>
template<typename PolicyType>
SGDR<
    DecomposableFunctionType,
    UpdatePolicyType,
    DecayPolicyType
>::SGDR(DecomposableFunctionType& function,
        const size_t epochRestart,
        const double multFactor,
        const size_t batchSize,
        const double stepSize,
        const size_t maxIterations,
        const double tolerance,
        const bool shuffle,
        const UpdatePolicyType& updatePolicy,
        const typename std::enable_if_t<std::is_same<
            PolicyType, CyclicalDecay>::value>* /* junk */) :
    function(function),
    batchSize(batchSize),
    decayPolicy(CyclicalDecay(epochRestart,
                              multFactor,
                              stepSize,
                              batchSize,
                              function.NumFunctions())),
    optimizer(OptimizerType(function,
                            batchSize,
                            stepSize,
                            maxIterations,
                            tolerance,
                            shuffle,
                            updatePolicy,
                            decayPolicy))
{
  /* Nothing to do here */
}

template<
    typename DecomposableFunctionType,
    typename UpdatePolicyType,
    typename DecayPolicyType
>
template<typename PolicyType>
double SGDR<
    DecomposableFunctionType,
    UpdatePolicyType,
    DecayPolicyType
>::Optimize(arma::mat& iterate,
            const bool accumulate,
            const typename std::enable_if_t<std::is_same<
                PolicyType, SnapshotEnsembles>::value>* /* junk */)
{
  // If a user changed the step size he hasn't update the step size of the
  // cyclical decay instantiation, so we have to do here.
  if (optimizer.StepSize() != decayPolicy.StepSize())
  {
    decayPolicy.StepSize() = optimizer.StepSize();
  }

  // If a user changed the batch size we have to update the restart fraction
  // of the cyclical decay instantiation.
  if (optimizer.BatchSize() != batchSize)
  {
    batchSize = optimizer.BatchSize();
    decayPolicy.EpochBatches() = function.NumFunctions() /
        double(batchSize);
  }

  double overallObjective = optimizer.Optimize(iterate);

  // Accumulate snapshots.
  if (accumulate)
  {
    for (size_t i = 0; i < decayPolicy.Snapshots().size(); ++i)
    {
      iterate += decayPolicy.Snapshots()[i];
    }
    iterate /= (decayPolicy.Snapshots().size() + 1);

    // Calculate final objective.
    overallObjective = 0;
    for (size_t i = 0; i < function.NumFunctions(); ++i)
      overallObjective += function.Evaluate(iterate, i);
  }

  return overallObjective;
}

template<
    typename DecomposableFunctionType,
    typename UpdatePolicyType,
    typename DecayPolicyType
>
template<typename PolicyType>
double SGDR<
    DecomposableFunctionType,
    UpdatePolicyType,
    DecayPolicyType
>::Optimize(arma::mat& iterate,
            const typename std::enable_if_t<std::is_same<
                PolicyType, CyclicalDecay>::value>* /* junk */)
{
  // If a user changed the step size he hasn't update the step size of the
  // cyclical decay instantiation, so we have to do here.
  if (optimizer.StepSize() != decayPolicy.StepSize())
  {
    decayPolicy.StepSize() = optimizer.StepSize();
  }

  // If a user changed the batch size we have to update the restart fraction
  // of the cyclical decay instantiation.
  if (optimizer.BatchSize() != batchSize)
  {
    batchSize = optimizer.BatchSize();
    decayPolicy.EpochBatches() = function.NumFunctions() /
        double(batchSize);
  }

  return optimizer.Optimize(iterate);
}

} // namespace optimization
} // namespace mlpack

#endif
