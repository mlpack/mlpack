/**
 * @file lmnn_function_impl.cpp
 * @author Manish Kumar
 *
 * An implementation of the LMNNFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_LMNN_FUNCTION_IMPL_HPP

#include "lmnn_function.hpp"
#include "constraints.hpp"

#include <mlpack/core/math/make_alias.hpp>
#include <mlpack/core/optimizers/function.hpp>

namespace mlpack {
namespace lmnn {

template<typename MetricType>
LMNNFunction<MetricType>::LMNNFunction(const arma::mat& dataset,
                           const arma::Row<size_t>& labels,
                           size_t k,
                           double regularization,
                           MetricType metric) :
    dataset(math::MakeAlias(const_cast<arma::mat&>(dataset), false)),
    labels(math::MakeAlias(const_cast<arma::Row<size_t>&>(labels), false)),
    k(k),
    metric(metric),
    regularization(regularization)
{
  // Initialize the initial learning point.
  initialPoint.eye(dataset.n_rows, dataset.n_rows);

  // Initialize target neighbors.
  Constraints constraint(dataset, labels, k);
  constraint.TargetNeighbors(targetNeighbors);
}

//! Shuffle the dataset.
template<typename MetricType>
void LMNNFunction<MetricType>::Shuffle()
{
  arma::mat newDataset;
  arma::Row<size_t> newLabels;

  math::ShuffleData(dataset, labels, newDataset, newLabels);

  math::ClearAlias(dataset);
  math::ClearAlias(labels);

  dataset = std::move(newDataset);
  labels = std::move(newLabels);
}

//! Calculate outer product of vector.
arma::mat OuterProduct(arma::vec& diff)
{
  return diff * arma::trans(diff);
}

//! Evaluate cost over whole dataset.
template<typename MetricType>
double LMNNFunction<MetricType>::Evaluate(const arma::mat& coordinates)
{
  double cost = 0;

  // Apply metric over dataset.
  transformedDataset = coordinates * dataset;

  // Re-calculate impostors on transformed dataset.
  Constraints constraint(transformedDataset, labels, k);
  constraint.Impostors(impostors);

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    for (size_t j = 0; j < k ; j++)
    {
      // Calculate cost due to distance between target neighbors & data point.
      double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;
    }

    for (size_t j = 0; j < k ; j++)
    {
      for (size_t l = 0; l < k ; l++)
      {
        // Calculate cost due to data point, target neighbors, impostors
        // triplets.
         double eval = metric.Evaluate(transformedDataset.col(i),
                            transformedDataset.col(targetNeighbors(j, i))) -
                       metric.Evaluate(transformedDataset.col(i),
                            transformedDataset.col(impostors(l, i)));

        cost += regularization * std::max(0.0, 1 + eval);
      }
    }
  }

  return cost;
};

//! Calculate cost over batches.
template<typename MetricType>
double LMNNFunction<MetricType>::Evaluate(const arma::mat& coordinates,
                                          const size_t begin,
                                          const size_t batchSize)
{
  double cost = 0;

  // Apply metric over dataset.
  transformedDataset = coordinates * dataset;

  Constraints constraint(transformedDataset, labels, k);
  constraint.Impostors(impostors);

  for (size_t i = begin; i < begin + batchSize; i++)
  {
    for (size_t j = 0; j < k ; j++)
    {
      // Calculate cost due to distance between target neighbors & data point.
      double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;
    }

    for (size_t j = 0; j < k ; j++)
    {
      for (size_t l = 0; l < k ; l++)
      {
        // Calculate cost due to data point, target neighbors, impostors
        // triplets.
         double eval = metric.Evaluate(transformedDataset.col(i),
                            transformedDataset.col(targetNeighbors(j, i))) -
                       metric.Evaluate(transformedDataset.col(i),
                            transformedDataset.col(impostors(l, i)));

        cost += regularization * std::max(0.0, 1 + eval);
      }
    }
  }

  return cost;
}

//! Compute gradient over whole dataset.
template<typename MetricType>
void LMNNFunction<MetricType>::Gradient(const arma::mat& coordinates,
                                        arma::mat& gradient)
{
  gradient.zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_rows; i++)
  {
    for (size_t j = 0; j < k ; j++)
    {
      // Calculate gradient due to target neighbors.
      arma::vec cij = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      gradient += (1 - regularization) * OuterProduct(cij);
    }

    for (size_t j = 0; j < k ; j++)
    {
      for (size_t l = 0; l < k ; l++)
      {
        // Calculate gradient due to triplets.
        arma::vec cij = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        arma::vec cil = dataset.col(i) - dataset.col(impostors(l, i));

        gradient += regularization * (OuterProduct(cij) - OuterProduct(cil));
      }
    }
  }
}

//! Compute gradient over a batch of data points.
template<typename MetricType>
template <typename GradType>
void LMNNFunction<MetricType>::Gradient(const arma::mat& coordinates,
                                        const size_t begin,
                                        GradType& gradient,
                                        const size_t batchSize)
{
  gradient.zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; i++)
  {
    for (size_t j = 0; j < k ; j++)
    {
      // Calculate gradient due to target neighbors.
      arma::vec cij = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      gradient += (1 - regularization) * OuterProduct(cij);
    }

    for (size_t j = 0; j < k ; j++)
    {
      for (size_t l = 0; l < k ; l++)
      {
        // Calculate gradient due to triplets.
        arma::vec cij = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        arma::vec cil = dataset.col(i) - dataset.col(impostors(l, i));

        gradient += regularization * (OuterProduct(cij) - OuterProduct(cil));
      }
    }
  }
}

} // namespace lmnn
} // namespace mlpack

namespace mlpack {
namespace optimization{

inline void Projection(arma::mat& iterate)
{
  // Projection.
  arma::cx_vec eigval;
  arma::cx_mat eigvec;

  arma::eig_gen(eigval, eigvec, iterate);

  arma::vec realEigVal = arma::real(eigval);
  arma::mat realEigVec = arma::real(eigvec);

  arma::uvec ind = arma::find(realEigVal > 0);
  arma::mat diagEigVal = arma::diagmat(realEigVal);
  iterate = realEigVec.cols(ind) * diagEigVal.submat(ind, ind) *
      arma::trans(realEigVec.cols(ind));
}

// Template specialization for the SGD optimizer.
template<>
template<typename DecomposableFunctionType>
double StandardSGD::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  typedef Function<DecomposableFunctionType> FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // Make sure we have all the methods that we need.
  traits::CheckDecomposableFunctionTypeAPI<FullFunctionType>();

  // Find the number of functions to use.
  const size_t numFunctions = f.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  // Initialize the update policy.
  if (resetPolicy)
    updatePolicy.Initialize(iterate.n_rows, iterate.n_cols);

  // Now iterate!
  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0 && i > 0)
    {
      // Output current objective function.
      Log::Info << "SGD: iteration " << i << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Log::Warn << "SGD: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Log::Info << "SGD: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        f.Shuffle();
    }

    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    const size_t effectiveBatchSize = std::min(
        std::min(batchSize, actualMaxIterations - i),
        numFunctions - currentFunction);

    // Technically we are computing the objective before we take the step, but
    // for many FunctionTypes it may be much quicker to do it like this.
    overallObjective += f.EvaluateWithGradient(iterate, currentFunction,
        gradient, effectiveBatchSize);

    // Apply projection.
    if (!currentFunction)
      Projection(iterate);

    // Use the update policy to take a step.
    updatePolicy.Update(iterate, stepSize, gradient);

    // Now update the learning rate if requested by the user.
    decayPolicy.Update(iterate, stepSize, gradient);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;
  }

  Log::Info << "SGD: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective.
  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    overallObjective += f.Evaluate(iterate, i, effectiveBatchSize);
  }
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
