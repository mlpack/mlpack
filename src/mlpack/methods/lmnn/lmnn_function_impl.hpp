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
    regularization(regularization),
    precalculated(false)
{
  // Initialize the initial learning point.
  initialPoint.eye(dataset.n_rows, dataset.n_rows);
  // Initialize transformed dataset to base dataset.
  transformedDataset = dataset;

  // Initialize target neighbors & impostors.
  Constraints constraint(dataset, labels, k);
  constraint.TargetNeighbors(targetNeighbors);
  constraint.Impostors(impostors);
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

  // Re-calculate target neighbors as indices changed.
  Constraints constraint(dataset, labels, k);
  constraint.TargetNeighbors(targetNeighbors);
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

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation. Here bp stands for
      // breaking point.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to data point, target neighbors, impostors
        // triplets.
        double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i))) -
                      metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(impostors(l, i)));

        // Check bounding condition.
        if (eval < -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);
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
  constraint.Impostors(impostors, begin, batchSize);

  for (size_t i = begin; i < begin + batchSize; i++)
  {
    for (size_t j = 0; j < k ; j++)
    {
      // Calculate cost due to distance between target neighbors & data point.
      double eval = metric.Evaluate(transformedDataset.col(i),
                        transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to data point, target neighbors, impostors
        // triplets.
        double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i))) -
                      metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(impostors(l, i)));

        // Check bounding condition.
        if (eval < -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);
      }
    }
  }

  return cost;
}

//! Compute gradient over whole dataset.
template<typename MetricType>
template<typename GradType>
void LMNNFunction<MetricType>::Gradient(const arma::mat& coordinates,
                                        GradType& gradient)
{
  gradient.zeros(coordinates.n_rows, coordinates.n_cols);

  // Calculate gradient due to target neighbors.
  Precalculate(dataset.n_cols);
  arma::mat cij = p_cij[0];

  // Calculate gradient due to impostors.
  arma::mat cil = arma::zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate gradient due to triplets.
        double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i))) -
                      metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(impostors(l, i)));

        // Check bounding condition.
        if (eval < -1)
        {
          // update bound.
          bp = l;
          break;
        }

        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * arma::trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * arma::trans(diff);
      }
    }
  }

  gradient = 2 * coordinates * ((1 - regularization) * cij +
      regularization * cil);
}

//! Compute gradient over a batch of data points.
template<typename MetricType>
template<typename GradType>
void LMNNFunction<MetricType>::Gradient(const arma::mat& coordinates,
                                        const size_t begin,
                                        GradType& gradient,
                                        const size_t batchSize)
{
  gradient.zeros(coordinates.n_rows, coordinates.n_cols);

  // Calculate gradient due to target neighbors.
  Precalculate(batchSize);
  arma::mat cij = p_cij[begin / batchSize];

  // Calculate gradient due to impostors.
  arma::mat cil = arma::zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; i++)
  {
    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate gradient due to triplets.
        double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i))) -
                      metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(impostors(l, i)));

        // Check bounding condition.
        if (eval < -1)
        {
          // update bound.
          bp = l;
          break;
        }

        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * arma::trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * arma::trans(diff);
      }
    }
  }

  gradient = 2 * coordinates * ((1 - regularization) * cij +
      regularization * cil);
}

//! Compute cost & gradient over whole dataset.
template<typename MetricType>
template<typename GradType>
double LMNNFunction<MetricType>::EvaluateWithGradient(
                                   const arma::mat& coordinates,
                                   GradType& gradient)
{
  double cost = 0;

  // Apply metric over dataset.
  transformedDataset = coordinates * dataset;

  // Calculate impostors.
  Constraints constraint(transformedDataset, labels, k);
  constraint.Impostors(impostors);

  gradient.zeros(coordinates.n_rows, coordinates.n_cols);

  // Calculate gradient due to target neighbors.
  Precalculate(dataset.n_cols);
  arma::mat cij = p_cij[0];

  // Calculate gradient due to impostors.
  arma::mat cil = arma::zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      // Calculate cost due to distance between target neighbors & data point.
      double eval = metric.Evaluate(transformedDataset.col(i),
                        transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i))) -
                      metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(impostors(l, i)));

        // Check bounding condition.
        if (eval < -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);

        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * arma::trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * arma::trans(diff);
      }
    }
  }

  gradient = 2 * coordinates * ((1 - regularization) * cij +
      regularization * cil);

  return cost;
}

//! Compute cost & gradient over a batch of data points.
template<typename MetricType>
template<typename GradType>
double LMNNFunction<MetricType>::EvaluateWithGradient(
                                   const arma::mat& coordinates,
                                   const size_t begin,
                                   GradType& gradient,
                                   const size_t batchSize)
{
  double cost = 0;

  // Apply metric over dataset.
  transformedDataset = coordinates * dataset;

  // Calculate impostors.
  Constraints constraint(transformedDataset, labels, k);
  constraint.Impostors(impostors, begin, batchSize);

  gradient.zeros(coordinates.n_rows, coordinates.n_cols);

  // Calculate gradient due to target neighbors.
  Precalculate(batchSize);
  arma::mat cij = p_cij[begin / batchSize];

  // Calculate gradient due to impostors.
  arma::mat cil = arma::zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; i++)
  {
    for (size_t j = 0; j < k ; j++)
    {
      // Calculate cost due to distance between target neighbors & data point.
      double eval = metric.Evaluate(transformedDataset.col(i),
                        transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(targetNeighbors(j, i))) -
                      metric.Evaluate(transformedDataset.col(i),
                          transformedDataset.col(impostors(l, i)));

        // Check bounding condition.
        if (eval < -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);

        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * arma::trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * arma::trans(diff);
      }
    }
  }

  gradient = 2 * coordinates * ((1 - regularization) * cij +
      regularization * cil);

  return cost;
}

template<typename MetricType>
void LMNNFunction<MetricType>::Precalculate(const size_t batchSize)
{
  // Make sure the calculation is necessary.
  if (precalculated)
    return;

  size_t numBatches = std::ceil(double(dataset.n_cols / batchSize));

  p_cij.resize(numBatches);

  for (size_t i = 0; i < numBatches; i++)
  {
    // Initialize p_cij as zero matrix.
    p_cij[i].zeros(dataset.n_rows, dataset.n_rows);

    for (size_t j = i * batchSize; j < (i + 1) * batchSize &&
        j < dataset.n_cols; j++)
    {
      for (size_t l = 0; l < k ; l++)
      {
        // Calculate gradient due to target neighbors.
        arma::vec diff = dataset.col(j) -
            dataset.col(targetNeighbors(l, j));
        p_cij[i] += diff * arma::trans(diff);
      }
    }
  }

  // We've done a precalculation.  Mark it as done.
  precalculated = true;
}

} // namespace lmnn
} // namespace mlpack

#endif
