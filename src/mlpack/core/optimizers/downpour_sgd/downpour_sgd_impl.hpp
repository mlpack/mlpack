/**
 * @file downpour_sgd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of Downpour SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_DOWNPOUR_SGD_DOWNPOUR_SGD_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_DOWNPOUR_SGD_DOWNPOUR_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "downpour_sgd.hpp"

#ifdef HAS_MPI
  #include <boost/mpi.hpp>
#endif

namespace mlpack {
namespace optimization {

template<typename UpdatePolicyType, typename DecayPolicyType>
DownpourSGD<UpdatePolicyType, DecayPolicyType>::DownpourSGD(
    const size_t batchSize,
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const size_t fetchSize,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy,
    const DecayPolicyType& decayPolicy,
    const bool resetPolicy) :
    batchSize(batchSize),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    fetchSize(fetchSize),
    shuffle(shuffle),
    updatePolicy(updatePolicy),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
#ifdef HAS_MPI
template<typename UpdatePolicyType, typename DecayPolicyType>
template<typename DecomposableFunctionType>
double DownpourSGD<UpdatePolicyType, DecayPolicyType>::Optimize(
  DecomposableFunctionType& function, arma::mat& iterate)
{
  // To keep track of where we are and how things are going.
  double currentObjectiv = 0;
  double overallObjective = 0;
  size_t currentBatch = 0;

  // Convenience typedef for the message tags.
  enum messageTags {batch, parameter, gradient, objective, finish};

  // Initialize the MPI environment.
  static boost::mpi::environment env;
  boost::mpi::communicator world;

  // Find the number of functions.
  const size_t numFunctions = function.NumFunctions();
  size_t numBatches = numFunctions / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  arma::uvec visitationOrder = arma::linspace<arma::uvec>(0,
      (numBatches - 1), numBatches);

  if (shuffle)
    visitationOrder = arma::shuffle(visitationOrder);

  const size_t workerSize = std::min((size_t) world.size(), numBatches + 1);

  // If the overall number of processes is greater than the number of batches
  // we have to split the the process into two groups, where the local group
  // performs the actual tasks.
  boost::mpi::communicator local = world.split(
      (size_t) world.rank() < workerSize ? 0 : 1);

  if (world.rank() == 0)
  {
    // Send parameters to all worker in the local group.
    boost::mpi::broadcast(local, iterate, 0);

    // To keep track of where we are and how things are going.
    double lastObjective = DBL_MAX;

    // Send batches to each worker.
    for (size_t w = 1; w < workerSize; ++w, currentBatch += fetchSize)
      world.send(w, messageTags::batch, currentBatch);

    // Now iterate!
    arma::mat gradient;
    for (size_t i = 1; i != maxIterations; ++i, currentBatch += fetchSize)
    {
      if ((currentBatch % numBatches) == 0)
      {
        // Output current objective function.
        Log::Info << "Downpour SGD: iteration " << i << ", objective "
            << overallObjective << "." << std::endl;

        if (std::isnan(overallObjective) || std::isinf(overallObjective))
        {
          Log::Warn << "Downpour SGD: converged to " << overallObjective
              << "; terminating with failure.  Try a smaller step size?"
              << std::endl;
          break;
        }

        if (std::abs(lastObjective - overallObjective) < tolerance)
        {
          Log::Info << "Downpour SGD: minimized within tolerance " << tolerance
              << "; terminating optimization." << std::endl;
          break;
        }

        // Reset the counter variables.
        lastObjective = overallObjective;
        overallObjective = 0;
        currentBatch = 0;

        if (shuffle)
          visitationOrder = arma::shuffle(visitationOrder);
      }

      // Get the message tag and perform the required task.
      boost::mpi::status msg = world.probe();

      // Fetch calclated gradient.
      world.recv(msg.source(), messageTags::gradient, gradient);

      // Now update the iterate.
      updatePolicy.Update(iterate, stepSize, gradient);

      // Fetch current objective and update overall objective.
      world.recv(msg.source(), messageTags::objective, currentObjectiv);
      overallObjective += currentObjectiv;

      // Now update the learning rate if requested by the user.
      decayPolicy.Update(iterate, stepSize, gradient);

      // Push updated parameter and new task.
      world.send(msg.source(), messageTags::parameter, iterate);
      world.send(msg.source(), messageTags::batch,
          visitationOrder[currentBatch]);
    }

    for (size_t w = 1; w < workerSize; ++w)
      world.send(w, messageTags::finish, 0);
  }
  else
  {
    if ((size_t) world.rank() < workerSize)
    {
      // Get initial parameter.
      boost::mpi::broadcast(local, iterate, 0);

      // Now iterate!
      arma::mat gradient(iterate.n_rows, iterate.n_cols);
      for (size_t i = 1;
          i != (maxIterations == 0 ? 0 : maxIterations * 2 + 10); ++i)
      {
        // Get the message tag and perform the requested task.
        boost::mpi::status msg = world.probe();
        if (msg.tag() == messageTags::batch)
        {
          // Fetch batch information.
          world.recv(0, boost::mpi::any_tag, currentBatch);
        }
        else if (msg.tag() == messageTags::parameter)
        {
          // Fetch parameter.
          world.recv(0, boost::mpi::any_tag, iterate);
          continue;
        }
        else if (msg.tag() == messageTags::finish)
        {
          // Optimization process finished.
          size_t finish;
          world.recv(0, messageTags::finish, finish);
          break;
        }
        else
        {
          continue;
        }

        const size_t numOffset = currentBatch + fetchSize - 1 < numBatches ?
            fetchSize : numBatches;
        for (size_t batchOffset = 0; batchOffset < numOffset; ++batchOffset)
        {
          // Evaluate the gradient for this mini-batch.
          const size_t offset = batchSize *
              visitationOrder[currentBatch + batchOffset];
          function.Gradient(iterate, offset, gradient);
          if (visitationOrder[currentBatch] != numBatches - 1)
          {
            for (size_t j = 1; j < batchSize; ++j)
            {
              arma::mat funcGradient;
              function.Gradient(iterate, offset + j, funcGradient);
              gradient += funcGradient;
            }

            gradient /= batchSize;

            // Add that to the current objective function.
            for (size_t j = 0; j < batchSize; ++j)
              currentObjectiv += function.Evaluate(iterate, offset + j);
          }
          else
          {
            // Handle last batch differently: it's not a full-size batch.
            const size_t lastBatchSize = numFunctions - offset - 1;
            for (size_t j = 1; j < lastBatchSize; ++j)
            {
              arma::mat funcGradient;
              function.Gradient(iterate, offset + j, funcGradient);
              gradient += funcGradient;
            }

            // Ensure the last batch size isn't zero, to avoid division by zero
            // before updating.
            if (lastBatchSize > 0)
              gradient /= lastBatchSize;

            // Add that to the overall objective function.
            for (size_t j = 0; j < lastBatchSize; ++j)
              currentObjectiv += function.Evaluate(iterate, offset + j);
          }
        }

        // Push calculated gradients and objective.
        world.isend(0, messageTags::gradient, gradient);
        world.isend(0, messageTags::objective, currentObjectiv);
      }
    }
  }

  // Wait for all processes within the local communicator to reach the barrier.
  local.barrier();

  return overallObjective;
}
#else
template<typename UpdatePolicyType, typename DecayPolicyType>
template<typename DecomposableFunctionType>
double DownpourSGD<UpdatePolicyType, DecayPolicyType>::Optimize(
  DecomposableFunctionType& /* function */, arma::mat& /* iterate */)
{
  return 0;
}
#endif

} // namespace optimization
} // namespace mlpack

#endif
