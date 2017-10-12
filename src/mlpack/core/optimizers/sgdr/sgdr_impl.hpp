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

template<typename UpdatePolicyType>
SGDR<UpdatePolicyType>::SGDR(
    const size_t epochRestart,
    const double multFactor,
    const size_t batchSize,
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy) :
    batchSize(batchSize),
    optimizer(OptimizerType(stepSize,
                            batchSize,
                            maxIterations,
                            tolerance,
                            shuffle,
                            updatePolicy,
                            CyclicalDecay(
                                epochRestart,
                                multFactor,
                                stepSize)))
{
  /* Nothing to do here */
}

template<typename UpdatePolicyType>
template<typename DecomposableFunctionType>
double SGDR<UpdatePolicyType>::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  // If a user changed the step size he hasn't update the step size of the
  // cyclical decay instantiation, so we have to do it here.
  if (optimizer.StepSize() != optimizer.DecayPolicy().StepSize())
  {
    optimizer.DecayPolicy().StepSize() = optimizer.StepSize();
  }

  optimizer.DecayPolicy().EpochBatches() = function.NumFunctions() /
      double(optimizer.BatchSize());

  // If a user changed the batch size we have to update the restart fraction
  // of the cyclical decay instantiation.
  if (optimizer.BatchSize() != batchSize)
  {
    batchSize = optimizer.BatchSize();
  }

  return optimizer.Optimize(function, iterate);
}

} // namespace optimization
} // namespace mlpack

#endif
