/**
 * @file bigbatch_sgd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of big-batch SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BIGBATCH_SGD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_BIGBATCH_SGD_BIGBATCH_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "bigbatch_sgd.hpp"

namespace mlpack {
namespace optimization {

template<typename UpdatePolicyType>
BigBatchSGD<UpdatePolicyType>::BigBatchSGD(
    const size_t batchSize,
    const double stepSize,
    const double batchDelta,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle) :
    batchSize(batchSize),
    stepSize(stepSize),
    batchDelta(batchDelta),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    updatePolicy(UpdatePolicyType())
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType>
template<typename DecomposableFunctionType>
double BigBatchSGD<UpdatePolicyType>::Optimize(
    DecomposableFunctionType& function, arma::mat& iterate)
{
  // Find the number of functions.
  const size_t numFunctions = function.NumFunctions();
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  // Batch visitation order.
  arma::Col<size_t> visitationOrder = arma::linspace<arma::Col<size_t>>(0,
      (numBatches - 1), numBatches);

  if (shuffle)
    visitationOrder = arma::shuffle(visitationOrder);

  // To keep track of where we are and how things are going.
  std::vector<arma::mat> funcGradients(batchSize);
  size_t currentBatch = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;
  double vB = 0;
  double gB = 0;
  bool reset = false;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  for (size_t i = 1; i != maxIterations; ++i, ++currentBatch)
  {
    // Is this iteration the start of a sequence?
    if ((currentBatch % numBatches) == 0)
    {
      // Output current objective function.
      Log::Info << "Big-batch SGD: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "Big-batch SGD: converged to " << overallObjective
            << "; terminating with failure.  Try a smaller step size?"
            << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "Big-batch SGD: minimized within tolerance " << tolerance
            << "; terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentBatch = 0;

      if (shuffle)
        visitationOrder = arma::shuffle(visitationOrder);
    }

    const size_t offset = batchSize * visitationOrder[currentBatch];
    if (visitationOrder[currentBatch] != numBatches - 1)
    {
      // Compute the stochastic gradient estimation.
      function.Gradient(iterate, offset, funcGradients[0]);
      gradient = funcGradients[0];
      for (size_t j = 1; j < batchSize; ++j)
      {
        function.Gradient(iterate, offset + j, funcGradients[j]);
        gradient += funcGradients[j];
      }
      gB = std::pow(arma::norm(gradient / batchSize, 2), 2.0);

      // Compute sample variance.
      vB = std::pow(arma::norm(funcGradients[0] - (gradient / batchSize),
          2.0), 2.0);
      for (size_t j = 1; j < batchSize; ++j)
      {
        vB += std::pow(arma::norm(funcGradients[j] - (gradient / batchSize),
            2.0), 2.0);
      }

      // Reset the batch size update process counter.
      reset = false;

      // Update batch size.
      while (gB <= ((1 / ((double) batchSize - 1) * vB) / batchSize))
      {
        // Increase batch size at least by one.
        size_t batchOffset = batchDelta * batchSize;
        if (batchOffset <= 0)
          batchOffset = 1;

        funcGradients.resize(batchSize + batchOffset);

        // Generate new batch indices.
        arma::Col<size_t> batchVisitationOrder;
        if ((offset + batchSize + batchOffset - 1) < numFunctions)
        {
          batchVisitationOrder = arma::linspace<arma::Col<size_t>>(offset +
              batchSize, (offset + batchSize + batchOffset - 1), batchOffset);
        }
        else if (((int) offset - (int) batchOffset) >= 0)
        {
          batchVisitationOrder = arma::linspace<arma::Col<size_t>>(offset -
              batchOffset, offset - 1, batchOffset);
        }
        else
        {
          batchVisitationOrder = arma::randi<arma::Col<size_t> >(batchOffset,
              arma::distr_param(0, numFunctions - 1));
        }

        // Update the stochastic gradient estimation.
        for (size_t j = 0; j < batchOffset; ++j)
        {
          function.Gradient(iterate, batchVisitationOrder[j],
                funcGradients[batchSize + j]);
          gradient += funcGradients[batchSize + j];
        }
        gB = std::pow(arma::norm(gradient /
            (batchSize + batchOffset), 2), 2.0);

        // Update sample variance.
        for (size_t j = 0; j < batchOffset; ++j)
        {
          vB += std::pow(arma::norm(funcGradients[batchSize + j] - (gradient /
              (batchSize + batchOffset)), 2.0), 2.0);
        }

        batchSize = batchSize + batchOffset;

        // Reset the counter variable and visitation order.
        numBatches = numFunctions / batchSize;
        if (numFunctions % batchSize != 0)
          ++numBatches; // Capture last few.

        visitationOrder = arma::shuffle(
            arma::linspace<arma::Col<size_t>>(0, (numBatches - 1), numBatches));

        if ((currentBatch % numBatches) == 0 || currentBatch >= numBatches)
          currentBatch = 0;

        // Batch size updated.
        reset = true;
      }

      size_t backtrackingBatchSize = batchSize;
      if ((offset + batchSize) > numFunctions)
        backtrackingBatchSize = numFunctions - offset;

      updatePolicy.Update(function, stepSize, iterate, gradient, gB, vB, offset,
          batchSize, backtrackingBatchSize, reset);

      // Update the iterate.
      iterate -= stepSize * gradient;

      // Add that to the overall objective function.
      for (size_t j = 0; j < backtrackingBatchSize; ++j)
        overallObjective += function.Evaluate(iterate, offset + j);
    }
    else
    {
      // Handle last batch differently: it's not a full-size batch.
      const size_t lastBatchSize = numFunctions - offset - 1;
      function.Gradient(iterate, offset, gradient);
      for (size_t j = 1; j < lastBatchSize; ++j)
      {
        function.Gradient(iterate, offset + j, funcGradients[0]);
        gradient += funcGradients[0];
      }

      // Ensure the last batch size isn't zero, to avoid division by zero before
      // updating.
      if (lastBatchSize > 0)
      {
        // Now update the iterate.
        iterate -= (stepSize / lastBatchSize) * gradient;
      }
      else
      {
        // Now update the iterate.
        iterate -= stepSize * gradient;
      }

      // Add that to the overall objective function.
      for (size_t j = 0; j < lastBatchSize; ++j)
        overallObjective += function.Evaluate(iterate, offset + j);
    }
  }

  Log::Info << "Big-batch SGD: maximum iterations (" << maxIterations << ") "
      << "reached; terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
