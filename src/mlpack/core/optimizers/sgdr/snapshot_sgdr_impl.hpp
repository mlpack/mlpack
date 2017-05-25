/**
 * @file snapshots_sgdr_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of SGDR method using snapshots ensembles.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGDR_SNAPSHOT_SGDR_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SGDR_SNAPSHOT_SGDR_IMPL_HPP

// In case it hasn't been included yet.
#include "snapshot_sgdr.hpp"

namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType, typename UpdatePolicyType>
SnapshotSGDR<DecomposableFunctionType, UpdatePolicyType>::SnapshotSGDR(
    DecomposableFunctionType& function,
    const size_t epochRestart,
    const double multFactor,
    const size_t batchSize,
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const size_t snapshots,
    const bool accumulate,
    const UpdatePolicyType& updatePolicy) :
    function(function),
    batchSize(batchSize),
    accumulate(accumulate),
    optimizer(OptimizerType(function,
                            batchSize,
                            stepSize,
                            maxIterations,
                            tolerance,
                            shuffle,
                            updatePolicy,
                            SnapshotEnsembles(
                                epochRestart,
                                multFactor,
                                stepSize,
                                batchSize,
                                function.NumFunctions(),
                                maxIterations,
                                snapshots)))
{
  /* Nothing to do here */
}

template<typename DecomposableFunctionType, typename UpdatePolicyType>
double SnapshotSGDR<DecomposableFunctionType, UpdatePolicyType>::Optimize(
    arma::mat& iterate)
{
  // If a user changed the step size he hasn't update the step size of the
  // cyclical decay instantiation, so we have to do here.
  if (optimizer.StepSize() != optimizer.DecayPolicy().StepSize())
  {
    optimizer.DecayPolicy().StepSize() = optimizer.StepSize();
  }

  // If a user changed the batch size we have to update the restart fraction
  // of the cyclical decay instantiation.
  if (optimizer.BatchSize() != batchSize)
  {
    batchSize = optimizer.BatchSize();
    optimizer.DecayPolicy().EpochBatches() = function.NumFunctions() /
        double(batchSize);
  }

  double overallObjective = optimizer.Optimize(iterate);

  // Accumulate snapshots.
  if (accumulate)
  {
    for (size_t i = 0; i < optimizer.DecayPolicy().Snapshots().size(); ++i)
    {
      iterate += optimizer.DecayPolicy().Snapshots()[i];
    }
    iterate /= (optimizer.DecayPolicy().Snapshots().size() + 1);

    // Calculate final objective.
    overallObjective = 0;
    for (size_t i = 0; i < function.NumFunctions(); ++i)
      overallObjective += function.Evaluate(iterate, i);
  }

  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
