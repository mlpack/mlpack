/**
 * @file methods/lmnn/lmnn_function_impl.hpp
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

#include <mlpack/core/math/make_alias.hpp>

namespace mlpack {

template<typename MetricType>
LMNNFunction<MetricType>::LMNNFunction(const arma::mat& datasetIn,
                                       const arma::Row<size_t>& labelsIn,
                                       size_t k,
                                       double regularization,
                                       size_t range,
                                       MetricType metric) :
    k(k),
    metric(metric),
    regularization(regularization),
    iteration(0),
    range(range),
    constraint(datasetIn, labelsIn, k),
    points(datasetIn.n_cols),
    impBounds(false)
{
  MakeAlias(dataset, datasetIn, datasetIn.n_rows, datasetIn.n_cols, false);
  MakeAlias(labels, labelsIn, labelsIn.n_rows, labelsIn.n_cols, false);

  // Initialize the initial learning point.
  initialPoint.eye(dataset.n_rows, dataset.n_rows);
  // Initialize transformed dataset to base dataset.
  transformedDataset = dataset;

  // Calculate and store norm of datapoints.
  norm.set_size(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    norm(i) = arma::norm(dataset.col(i));
  }

  // Initialize cache.
  evalOld.set_size(k, k, dataset.n_cols);
  evalOld.zeros();

  maxImpNorm.set_size(k, dataset.n_cols);
  maxImpNorm.zeros();

  lastTransformationIndices.set_size(dataset.n_cols);
  lastTransformationIndices.zeros();

  // Reserve the first element of cache.
  arma::mat emptyMat;
  oldTransformationMatrices.push_back(emptyMat);
  oldTransformationCounts.push_back(dataset.n_cols);

  // Check if we can impose bounds over impostors.
  size_t minCount = min(arma::histc(labels, arma::unique(labels)));
  if (minCount <= k + 1)
  {
    // Initialize target neighbors & impostors.
    targetNeighbors.set_size(k, dataset.n_cols);
    impostors.set_size(k, dataset.n_cols);
    distance.set_size(k, dataset.n_cols);
  }
  else
  {
    // Update parameters.
    constraint.K() = k + 1;
    impBounds = true;
    // Initialize target neighbors & impostors.
    targetNeighbors.set_size(k + 1, dataset.n_cols);
    impostors.set_size(k + 1, dataset.n_cols);
    distance.set_size(k + 1, dataset.n_cols);
  }

  constraint.TargetNeighbors(targetNeighbors, dataset, labels, norm);
  constraint.Impostors(impostors, dataset, labels, norm);

  // Precalculate and save the gradient due to target neighbors.
  Precalculate();
}

//! Shuffle the dataset.
template<typename MetricType>
void LMNNFunction<MetricType>::Shuffle()
{
  arma::mat newDataset = dataset;
  arma::Mat<size_t> newLabels = labels;
  arma::cube newEvalOld = evalOld;
  arma::vec newlastTransformationIndices = lastTransformationIndices;
  arma::mat newMaxImpNorm = maxImpNorm;
  arma::vec newNorm = norm;

  // Generate ordering.
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
      dataset.n_cols - 1, dataset.n_cols));

  ClearAlias(dataset);
  ClearAlias(labels);

  dataset = newDataset.cols(ordering);
  labels = newLabels.cols(ordering);
  maxImpNorm = newMaxImpNorm.cols(ordering);
  lastTransformationIndices = newlastTransformationIndices.elem(ordering);
  norm = newNorm.elem(ordering);

  for (size_t i = 0; i < ordering.n_elem; ++i)
  {
    evalOld.slice(i) = newEvalOld.slice(ordering(i));
  }

  // Re-calculate target neighbors as indices changed.
  constraint.PreCalulated() = false;
  constraint.TargetNeighbors(targetNeighbors, dataset, labels, norm);
}

// Update cache transformation matrices.
template<typename MetricType>
inline void LMNNFunction<MetricType>::UpdateCache(
                                          const arma::mat& transformation,
                                          const size_t begin,
                                          const size_t batchSize)
{
  // Are there any empty transformation matrices?
  size_t index = oldTransformationMatrices.size();
  for (size_t i = 1; i < oldTransformationCounts.size(); ++i)
  {
    if (oldTransformationCounts[i] == 0)
    {
      index = i; // Reuse this index.
      break;
    }
  }

  // Did we find an unused matrix?  If not, we have to allocate new space.
  if (index == oldTransformationMatrices.size())
  {
    oldTransformationMatrices.push_back(transformation);
    oldTransformationCounts.push_back(0);
  }
  else
  {
    oldTransformationMatrices[index] = transformation;
  }

  // Update all the transformation indices.
  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    --oldTransformationCounts[lastTransformationIndices(i)];
    lastTransformationIndices(i) = index;
  }

  oldTransformationCounts[index] += batchSize;

  #ifdef DEBUG
    size_t total = 0;
    for (size_t i = 1; i < oldTransformationCounts.size(); ++i)
    {
      std::ostringstream oss;
      oss << "transformation counts for matrix " << i
          << " invalid (" << oldTransformationCounts[i] << ")!";
      Log::Assert(oldTransformationCounts[i] <= dataset.n_cols, oss.str());
      total += oldTransformationCounts[i];
    }

    std::ostringstream oss;
    oss << "total count for transformation matrices invalid (" << total
        << ", " << "should be " << dataset.n_cols << "!";
    if (begin + batchSize == dataset.n_cols)
      Log::Assert(total == dataset.n_cols, oss.str());
  #endif
}

// Calculate norm of change in transformation.
template<typename MetricType>
inline void LMNNFunction<MetricType>::TransDiff(
                                std::map<size_t, double>& transformationDiffs,
                                const arma::mat& transformation,
                                const size_t begin,
                                const size_t batchSize)
{
  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    if (transformationDiffs.count(lastTransformationIndices[i]) == 0)
    {
      if (lastTransformationIndices[i] == 0)
      {
        transformationDiffs[0] = 0.0; // This won't be used anyway...
      }
      else
      {
        transformationDiffs[lastTransformationIndices[i]] =
            arma::norm(transformation -
            oldTransformationMatrices[lastTransformationIndices(i)]);
      }
    }
  }
}

//! Evaluate cost over whole dataset.
template<typename MetricType>
double LMNNFunction<MetricType>::Evaluate(const arma::mat& transformation)
{
  double cost = 0;

  // Apply metric over dataset.
  transformedDataset = transformation * dataset;

  double transformationDiff = 0;
  if (!transformationOld.is_empty())
  {
    // Calculate norm of change in transformation.
    transformationDiff = arma::norm(transformation - transformationOld);
  }

  if (!transformationOld.is_empty() && iteration++ % range == 0)
  {
    if (impBounds)
    {
      // Track number of data points to use for impostors calculatiom.
      size_t numPoints = 0;

      for (size_t i = 0; i < dataset.n_cols; ++i)
      {
        if (transformationDiff * (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distance(k, i) - distance(k - 1, i))
        {
          points(numPoints++) = i;
        }
      }

      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance,
          transformedDataset, labels, norm, points, numPoints);
    }
    else
    {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance, transformedDataset, labels,
          norm);
    }
  }
  else if (iteration++ % range == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance, transformedDataset, labels, norm);
  }

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
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
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = 0;

        // Bounds for eval.
        if (!transformationOld.is_empty() && evalOld(l, j, i) < -1)
        {
          // Update cache max impostor norm.
          maxImpNorm(l, i) = std::max(maxImpNorm(l, i), norm(impostors(l, i)));

          eval = evalOld(l, j, i) + transformationDiff *
              (norm(targetNeighbors(j, i)) + maxImpNorm(l, i) +
              2 * norm(i));
        }

        // Calculate exact eval value.
        if (eval > -1)
        {
          if (iteration - 1 % range == 0)
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distance(l, i);
          }
          else
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   metric.Evaluate(transformedDataset.col(i),
                       transformedDataset.col(impostors(l, i)));
          }
        }

        // Update cache eval value.
        evalOld(l, j, i) = eval;

        // Check bounding condition.
        if (eval <= -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);

        // Reset cache.
        if (eval > -1)
        {
          // update bound.
          evalOld(l, j, i) = 0;
          maxImpNorm(l, i) = 0;
        }
      }
    }
  }

  // Update cache transformation matrix.
  transformationOld = transformation;

  return cost;
}

//! Calculate cost over batches.
template<typename MetricType>
double LMNNFunction<MetricType>::Evaluate(const arma::mat& transformation,
                                          const size_t begin,
                                          const size_t batchSize)
{
  double cost = 0;

  // Calculate norm of change in transformation.
  std::map<size_t, double> transformationDiffs;
  TransDiff(transformationDiffs, transformation, begin, batchSize);

  // Apply metric over dataset.
  transformedDataset = transformation * dataset;

  if (impBounds && iteration++ % range == 0)
  {
    // Track number of data points to use for impostors calculatiom.
    size_t numPoints = 0;

    for (size_t i = begin; i < begin + batchSize; ++i)
    {
      if (lastTransformationIndices(i))
      {
        if (transformationDiffs[lastTransformationIndices[i]] *
            (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distance(k, i) - distance(k - 1, i))
        {
          points(numPoints++)  = i;
        }
      }
      else
      {
        points(numPoints++) = i;
      }
    }

    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance,
        transformedDataset, labels, norm, points, numPoints);
  }
  else if (iteration++ % range == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance, transformedDataset, labels,
        norm, begin, batchSize);
  }

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
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
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = 0;

        // Bounds for eval.
        if (lastTransformationIndices(i) && evalOld(l, j, i) < -1)
        {
          // Update cache max impostor norm.
          maxImpNorm(l, i) = std::max(maxImpNorm(l, i), norm(impostors(l, i)));

          eval = evalOld(l, j, i) +
              transformationDiffs[lastTransformationIndices[i]] *
              (norm(targetNeighbors(j, i)) + maxImpNorm(l, i) + 2 * norm(i));
        }

        // Calculate exact eval value.
        if (eval > -1)
        {
          if (iteration - 1 % range == 0)
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distance(l, i);
          }
          else
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   metric.Evaluate(transformedDataset.col(i),
                       transformedDataset.col(impostors(l, i)));
          }
        }

        // Update cache eval value.
        evalOld(l, j, i) = eval;

        // Check bounding condition.
        if (eval <= -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);

        // Reset cache.
        if (eval > -1 && lastTransformationIndices(i))
        {
          // update bound.
          evalOld(l, j, i) = 0;
          maxImpNorm(l, i) = 0;
          --oldTransformationCounts[lastTransformationIndices(i)];
          lastTransformationIndices(i) = 0;
        }
      }
    }
  }

  // Update cache.
  UpdateCache(transformation, begin, batchSize);

  return cost;
}

//! Compute gradient over whole dataset.
template<typename MetricType>
template<typename GradType>
void LMNNFunction<MetricType>::Gradient(const arma::mat& transformation,
                                        GradType& gradient)
{
  // Apply metric over dataset.
  transformedDataset = transformation * dataset;

  double transformationDiff = 0;
  if (!transformationOld.is_empty() && iteration++ % range == 0)
  {
    // Calculate norm of change in transformation.
    transformationDiff = arma::norm(transformation - transformationOld);

    if (impBounds)
    {
      // Track number of data points to use for impostors calculatiom.
      size_t numPoints = 0;

      for (size_t i = 0; i < dataset.n_cols; ++i)
      {
        if (transformationDiff * (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distance(k, i) - distance(k - 1, i))
        {
          points(numPoints++) = i;
        }
      }

      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance,
          transformedDataset, labels, norm, points, numPoints);
    }
    else
    {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance, transformedDataset, labels,
          norm);
    }
  }
  else if (iteration++ % range == 0)
  {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance, transformedDataset, labels,
          norm);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  // Calculate gradient due to target neighbors.
  arma::mat cij = pCij;

  // Calculate gradient due to impostors.
  arma::mat cil = zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = 0;

        // Bounds for eval.
        if (!transformationOld.is_empty() && evalOld(l, j, i) < -1)
        {
          // Update cache max impostor norm.
          maxImpNorm(l, i) = std::max(maxImpNorm(l, i), norm(impostors(l, i)));

          eval = evalOld(l, j, i) + transformationDiff *
              (norm(targetNeighbors(j, i)) + maxImpNorm(l, i) +
              2 * norm(i));
        }

        // Calculate exact eval value.
        if (eval > -1)
        {
          if (iteration - 1 % range == 0)
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distance(l, i);
          }
          else
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   metric.Evaluate(transformedDataset.col(i),
                       transformedDataset.col(impostors(l, i)));
          }
        }

        // Update cache eval value.
        evalOld(l, j, i) = eval;

        // Check bounding condition.
        if (eval <= -1)
        {
          // update bound.
          bp = l;
          break;
        }

        // Reset cache.
        if (eval > -1)
        {
          // update bound.
          evalOld(l, j, i) = 0;
          maxImpNorm(l, i) = 0;
        }

        // Caculate gradient due to impostors.
        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * trans(diff);
      }
    }
  }

  gradient = 2 * transformation * ((1 - regularization) * cij +
      regularization * cil);

  // Update cache transformation matrix.
  transformationOld = transformation;
}

//! Compute gradient over a batch of data points.
template<typename MetricType>
template<typename GradType>
void LMNNFunction<MetricType>::Gradient(const arma::mat& transformation,
                                        const size_t begin,
                                        GradType& gradient,
                                        const size_t batchSize)
{
  // Apply metric over dataset.
  transformedDataset = transformation * dataset;

  // Calculate norm of change in transformation.
  std::map<size_t, double> transformationDiffs;
  TransDiff(transformationDiffs, transformation, begin, batchSize);

  if (impBounds && iteration++ % range == 0)
  {
    // Track number of data points to use for impostors calculatiom.
    size_t numPoints = 0;

    for (size_t i = begin; i < begin + batchSize; ++i)
    {
      if (lastTransformationIndices(i))
      {
        if (transformationDiffs[lastTransformationIndices[i]] *
            (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distance(k, i) - distance(k - 1, i))
        {
          points(numPoints++)  = i;
        }
      }
      else
      {
        points(numPoints++) = i;
      }
    }

    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance,
        transformedDataset, labels, norm, points, numPoints);
  }
  else if (iteration++ % range == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance, transformedDataset, labels,
        norm, begin, batchSize);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  arma::mat cij = zeros(dataset.n_rows, dataset.n_rows);
  arma::mat cil = zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate gradient due to target neighbors.
      arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      cij += diff * trans(diff);
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = 0;

        // Bounds for eval.
        if (lastTransformationIndices(i) && evalOld(l, j, i) < -1)
        {
          // Update cache max impostor norm.
          maxImpNorm(l, i) = std::max(maxImpNorm(l, i), norm(impostors(l, i)));

          eval = evalOld(l, j, i) +
              transformationDiffs[lastTransformationIndices[i]] *
              (norm(targetNeighbors(j, i)) + maxImpNorm(l, i) + 2 * norm(i));
        }

        // Calculate exact eval value.
        if (eval > -1)
        {
          if (iteration - 1 % range == 0)
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distance(l, i);
          }
          else
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   metric.Evaluate(transformedDataset.col(i),
                       transformedDataset.col(impostors(l, i)));
          }
        }

        // Update cache eval value.
        evalOld(l, j, i) = eval;

        // Check bounding condition.
        if (eval <= -1)
        {
          // update bound.
          bp = l;
          break;
        }

        // Reset cache.
        if (eval > -1 && lastTransformationIndices(i))
        {
          // update bound.
          evalOld(l, j, i) = 0;
          maxImpNorm(l, i) = 0;
          --oldTransformationCounts[lastTransformationIndices(i)];
          lastTransformationIndices(i) = 0;
        }

        // Caculate gradient due to impostors.
        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * trans(diff);
      }
    }
  }

  gradient = 2 * transformation * ((1 - regularization) * cij +
      regularization * cil);

  // Update cache.
  UpdateCache(transformation, begin, batchSize);
}

//! Compute cost & gradient over whole dataset.
template<typename MetricType>
template<typename GradType>
double LMNNFunction<MetricType>::EvaluateWithGradient(
                                   const arma::mat& transformation,
                                   GradType& gradient)
{
  double cost = 0;

  // Apply metric over dataset.
  transformedDataset = transformation * dataset;

  double transformationDiff = 0;
  if (!transformationOld.is_empty())
  {
    // Calculate norm of change in transformation.
    transformationDiff = arma::norm(transformation - transformationOld);
  }

  if (!transformationOld.is_empty() && iteration++ % range == 0)
  {
    if (impBounds)
    {
      // Track number of data points to use for impostors calculatiom.
      size_t numPoints = 0;

      for (size_t i = 0; i < dataset.n_cols; ++i)
      {
        if (transformationDiff * (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distance(k, i) - distance(k - 1, i))
        {
          points(numPoints++) = i;
        }
      }

      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance,
          transformedDataset, labels, norm, points, numPoints);
    }
    else
    {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance, transformedDataset, labels,
          norm);
    }
  }
  else if (iteration++ % range == 0)
  {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distance, transformedDataset, labels,
          norm);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  // Calculate gradient due to target neighbors.
  arma::mat cij = pCij;

  // Calculate gradient due to impostors.
  arma::mat cil = zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
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
        double eval = 0;

        // Bounds for eval.
        if (!transformationOld.is_empty() && evalOld(l, j, i) < -1)
        {
          // Update cache max impostor norm.
          maxImpNorm(l, i) = std::max(maxImpNorm(l, i), norm(impostors(l, i)));

          eval = evalOld(l, j, i) + transformationDiff *
              (norm(targetNeighbors(j, i)) + maxImpNorm(l, i) +
              2 * norm(i));
        }

        // Calculate exact eval value.
        if (eval > -1)
        {
          if (iteration - 1 % range == 0)
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distance(l, i);
          }
          else
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   metric.Evaluate(transformedDataset.col(i),
                       transformedDataset.col(impostors(l, i)));
          }
        }

        // Update cache eval value.
        evalOld(l, j, i) = eval;

        // Check bounding condition.
        if (eval <= -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);

        // Caculate gradient due to impostors.
        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * trans(diff);
      }
    }
  }

  gradient = 2 * transformation * ((1 - regularization) * cij +
      regularization * cil);

  // Update cache transformation matrix.
  transformationOld = transformation;

  return cost;
}

//! Compute cost & gradient over a batch of data points.
template<typename MetricType>
template<typename GradType>
double LMNNFunction<MetricType>::EvaluateWithGradient(
                                   const arma::mat& transformation,
                                   const size_t begin,
                                   GradType& gradient,
                                   const size_t batchSize)
{
  double cost = 0;

  // Calculate norm of change in transformation.
  std::map<size_t, double> transformationDiffs;
  TransDiff(transformationDiffs, transformation, begin, batchSize);

  // Apply metric over dataset.
  transformedDataset = transformation * dataset;

  if (impBounds && iteration++ % range == 0)
  {
    // Track number of data points to use for impostors calculatiom.
    size_t numPoints = 0;

    for (size_t i = begin; i < begin + batchSize; ++i)
    {
      if (lastTransformationIndices(i))
      {
        if (transformationDiffs[lastTransformationIndices[i]] *
            (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distance(k, i) - distance(k - 1, i))
        {
          points(numPoints++)  = i;
        }
      }
      else
      {
        points(numPoints++) = i;
      }
    }

    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance,
        transformedDataset, labels, norm, points, numPoints);
  }
  else if (iteration++ % range == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distance, transformedDataset, labels,
        norm, begin, batchSize);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  arma::mat cij = zeros(dataset.n_rows, dataset.n_rows);
  arma::mat cil = zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate cost due to distance between target neighbors & data point.
      double eval = metric.Evaluate(transformedDataset.col(i),
                        transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;

      // Calculate gradient due to target neighbors.
      arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      cij += diff * trans(diff);
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        double eval = 0;

        // Bounds for eval.
        if (lastTransformationIndices(i) && evalOld(l, j, i) < -1)
        {
          // Update cache max impostor norm.
          maxImpNorm(l, i) = std::max(maxImpNorm(l, i), norm(impostors(l, i)));

          eval = evalOld(l, j, i) +
              transformationDiffs[lastTransformationIndices[i]] *
              (norm(targetNeighbors(j, i)) + maxImpNorm(l, i) + 2 * norm(i));
        }

        // Calculate exact eval value.
        if (eval > -1)
        {
          if (iteration - 1 % range == 0)
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distance(l, i);
          }
          else
          {
            eval = metric.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   metric.Evaluate(transformedDataset.col(i),
                       transformedDataset.col(impostors(l, i)));
          }
        }

        // Update cache eval value.
        evalOld(l, j, i) = eval;

        // Check bounding condition.
        if (eval <= -1)
        {
          // update bound.
          bp = l;
          break;
        }

        cost += regularization * (1 + eval);

        // Caculate gradient due to impostors.
        arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
        cil += diff * trans(diff);

        diff = dataset.col(i) - dataset.col(impostors(l, i));
        cil -= diff * trans(diff);
      }
    }
  }

  gradient = 2 * transformation * ((1 - regularization) * cij +
      regularization * cil);

  // Update cache.
  UpdateCache(transformation, begin, batchSize);

  return cost;
}

template<typename MetricType>
inline void LMNNFunction<MetricType>::Precalculate()
{
  pCij.zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate gradient due to target neighbors.
      arma::vec diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      pCij += diff * trans(diff);
    }
  }
}

} // namespace mlpack

#endif
