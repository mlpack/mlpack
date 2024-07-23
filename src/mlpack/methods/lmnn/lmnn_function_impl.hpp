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

template<typename MatType, typename LabelsType, typename DistanceType>
LMNNFunction<MatType, LabelsType, DistanceType>::LMNNFunction(
    const MatType& datasetIn,
    const LabelsType& labelsIn,
    size_t k,
    double regularization,
    size_t updateInterval,
    DistanceType distance) :
    k(k),
    distance(distance),
    regularization(regularization),
    iteration(0),
    updateInterval(updateInterval),
    constraint(datasetIn, labelsIn, k),
    points(datasetIn.n_cols),
    impBounds(false)
{
  MakeAlias(dataset, datasetIn, datasetIn.n_rows, datasetIn.n_cols, 0, false);
  MakeAlias(labels, labelsIn, labelsIn.n_elem, 0, false);

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
  MatType emptyMat;
  oldTransformationMatrices.push_back(emptyMat);
  oldTransformationCounts.push_back(dataset.n_cols);

  // Check if we can impose bounds over impostors.
  size_t minCount = min(arma::histc(labels, arma::unique(labels)));
  if (minCount <= k + 1)
  {
    // Initialize target neighbors & impostors.
    targetNeighbors.set_size(k, dataset.n_cols);
    impostors.set_size(k, dataset.n_cols);
    distanceMat.set_size(k, dataset.n_cols);
  }
  else
  {
    // Update parameters.
    constraint.K() = k + 1;
    impBounds = true;
    // Initialize target neighbors & impostors.
    targetNeighbors.set_size(k + 1, dataset.n_cols);
    impostors.set_size(k + 1, dataset.n_cols);
    distanceMat.set_size(k + 1, dataset.n_cols);
  }

  constraint.TargetNeighbors(targetNeighbors, dataset, labels, norm);
  constraint.Impostors(impostors, dataset, labels, norm);

  // Precalculate and save the gradient due to target neighbors.
  Precalculate();
}

//! Shuffle the dataset.
template<typename MatType, typename LabelsType, typename DistanceType>
void LMNNFunction<MatType, LabelsType, DistanceType>::Shuffle()
{
  MatType newDataset = dataset;
  LabelsType newLabels = labels;
  CubeType newEvalOld = evalOld;
  VecType newlastTransformationIndices = lastTransformationIndices;
  MatType newMaxImpNorm = maxImpNorm;
  VecType newNorm = norm;

  // Generate ordering.
  UVecType ordering = arma::shuffle(arma::linspace<UVecType>(0,
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
template<typename MatType, typename LabelsType, typename DistanceType>
inline void LMNNFunction<MatType, LabelsType, DistanceType>::UpdateCache(
    const MatType& transformation,
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
}

// Calculate norm of change in transformation.
template<typename MatType, typename LabelsType, typename DistanceType>
inline void LMNNFunction<MatType, LabelsType, DistanceType>::TransDiff(
    std::unordered_map<size_t, ElemType>& transformationDiffs,
    const MatType& transformation,
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
template<typename MatType, typename LabelsType, typename DistanceType>
typename MatType::elem_type
LMNNFunction<MatType, LabelsType, DistanceType>::Evaluate(
    const MatType& transformation)
{
  ElemType cost = 0;

  // Apply distance metric over dataset.
  transformedDataset = transformation * dataset;

  ElemType transformationDiff = 0;
  if (!transformationOld.is_empty())
  {
    // Calculate norm of change in transformation.
    transformationDiff = arma::norm(transformation - transformationOld);
  }

  if (!transformationOld.is_empty() && iteration++ % updateInterval == 0)
  {
    if (impBounds)
    {
      // Track number of data points to use for impostors calculatiom.
      size_t numPoints = 0;

      for (size_t i = 0; i < dataset.n_cols; ++i)
      {
        if (transformationDiff * (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distanceMat(k, i) - distanceMat(k - 1, i))
        {
          points(numPoints++) = i;
        }
      }

      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat,
          transformedDataset, labels, norm, points, numPoints);
    }
    else
    {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
          norm);
    }
  }
  else if (iteration++ % updateInterval == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
        norm);
  }

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate cost due to distance between target neighbors & data point.
      ElemType eval = distance.Evaluate(transformedDataset.col(i),
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
        ElemType eval = 0;

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
          if (iteration - 1 % updateInterval == 0)
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distanceMat(l, i);
          }
          else
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   distance.Evaluate(transformedDataset.col(i),
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
template<typename MatType, typename LabelsType, typename DistanceType>
typename MatType::elem_type
LMNNFunction<MatType, LabelsType, DistanceType>::Evaluate(
    const MatType& transformation,
    const size_t begin,
    const size_t batchSize)
{
  ElemType cost = 0;

  // Calculate norm of change in transformation.
  std::unordered_map<size_t, ElemType> transformationDiffs;
  TransDiff(transformationDiffs, transformation, begin, batchSize);

  // Apply distance metric over dataset.
  transformedDataset = transformation * dataset;

  if (impBounds && iteration++ % updateInterval == 0)
  {
    // Track number of data points to use for impostors calculatiom.
    size_t numPoints = 0;

    for (size_t i = begin; i < begin + batchSize; ++i)
    {
      if (lastTransformationIndices(i))
      {
        if (transformationDiffs[lastTransformationIndices[i]] *
            (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distanceMat(k, i) - distanceMat(k - 1, i))
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
    constraint.Impostors(impostors, distanceMat,
        transformedDataset, labels, norm, points, numPoints);
  }
  else if (iteration++ % updateInterval == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
        norm, begin, batchSize);
  }

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate cost due to distance between target neighbors & data point.
      ElemType eval = distance.Evaluate(transformedDataset.col(i),
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
        ElemType eval = 0;

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
          if (iteration - 1 % updateInterval == 0)
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distanceMat(l, i);
          }
          else
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   distance.Evaluate(transformedDataset.col(i),
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
template<typename MatType, typename LabelsType, typename DistanceType>
template<typename GradType>
void LMNNFunction<MatType, LabelsType, DistanceType>::Gradient(
    const MatType& transformation, GradType& gradient)
{
  // Apply distance metric over dataset.
  transformedDataset = transformation * dataset;

  ElemType transformationDiff = 0;
  if (!transformationOld.is_empty() && iteration++ % updateInterval == 0)
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
            norm(impostors(k, i))) > distanceMat(k, i) - distanceMat(k - 1, i))
        {
          points(numPoints++) = i;
        }
      }

      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat,
          transformedDataset, labels, norm, points, numPoints);
    }
    else
    {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
          norm);
    }
  }
  else if (iteration++ % updateInterval == 0)
  {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
          norm);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  // Calculate gradient due to target neighbors.
  MatType cij = pCij;

  // Calculate gradient due to impostors.
  MatType cil = zeros<MatType>(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        ElemType eval = 0;

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
          if (iteration - 1 % updateInterval == 0)
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distanceMat(l, i);
          }
          else
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   distance.Evaluate(transformedDataset.col(i),
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
        VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
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
template<typename MatType, typename LabelsType, typename DistanceType>
template<typename GradType>
void LMNNFunction<MatType, LabelsType, DistanceType>::Gradient(
    const MatType& transformation,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize)
{
  // Apply distance metric over dataset.
  transformedDataset = transformation * dataset;

  // Calculate norm of change in transformation.
  std::unordered_map<size_t, ElemType> transformationDiffs;
  TransDiff(transformationDiffs, transformation, begin, batchSize);

  if (impBounds && iteration++ % updateInterval == 0)
  {
    // Track number of data points to use for impostors calculatiom.
    size_t numPoints = 0;

    for (size_t i = begin; i < begin + batchSize; ++i)
    {
      if (lastTransformationIndices(i))
      {
        if (transformationDiffs[lastTransformationIndices[i]] *
            (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distanceMat(k, i) - distanceMat(k - 1, i))
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
    constraint.Impostors(impostors, distanceMat,
        transformedDataset, labels, norm, points, numPoints);
  }
  else if (iteration++ % updateInterval == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
        norm, begin, batchSize);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  MatType cij = zeros<MatType>(dataset.n_rows, dataset.n_rows);
  MatType cil = zeros<MatType>(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate gradient due to target neighbors.
      VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      cij += diff * trans(diff);
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        ElemType eval = 0;

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
          if (iteration - 1 % updateInterval == 0)
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distanceMat(l, i);
          }
          else
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   distance.Evaluate(transformedDataset.col(i),
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
        VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
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
template<typename MatType, typename LabelsType, typename DistanceType>
template<typename GradType>
typename MatType::elem_type
LMNNFunction<MatType, LabelsType, DistanceType>::EvaluateWithGradient(
    const MatType& transformation,
    GradType& gradient)
{
  ElemType cost = 0;

  // Apply distance metric over dataset.
  transformedDataset = transformation * dataset;

  ElemType transformationDiff = 0;
  if (!transformationOld.is_empty())
  {
    // Calculate norm of change in transformation.
    transformationDiff = arma::norm(transformation - transformationOld);
  }

  if (!transformationOld.is_empty() && iteration++ % updateInterval == 0)
  {
    if (impBounds)
    {
      // Track number of data points to use for impostors calculatiom.
      size_t numPoints = 0;

      for (size_t i = 0; i < dataset.n_cols; ++i)
      {
        if (transformationDiff * (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distanceMat(k, i) - distanceMat(k - 1, i))
        {
          points(numPoints++) = i;
        }
      }

      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat,
          transformedDataset, labels, norm, points, numPoints);
    }
    else
    {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
          norm);
    }
  }
  else if (iteration++ % updateInterval == 0)
  {
      // Re-calculate impostors on transformed dataset.
      constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
          norm);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  // Calculate gradient due to target neighbors.
  MatType cij = pCij;

  // Calculate gradient due to impostors.
  MatType cil = zeros<MatType>(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate cost due to distance between target neighbors & data point.
      ElemType eval = distance.Evaluate(transformedDataset.col(i),
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
        ElemType eval = 0;

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
          if (iteration - 1 % updateInterval == 0)
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distanceMat(l, i);
          }
          else
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   distance.Evaluate(transformedDataset.col(i),
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
        VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
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
template<typename MatType, typename LabelsType, typename DistanceType>
template<typename GradType>
typename MatType::elem_type
LMNNFunction<MatType, LabelsType, DistanceType>::EvaluateWithGradient(
    const MatType& transformation,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize)
{
  ElemType cost = 0;

  // Calculate norm of change in transformation.
  std::unordered_map<size_t, ElemType> transformationDiffs;
  TransDiff(transformationDiffs, transformation, begin, batchSize);

  // Apply distance metric over dataset.
  transformedDataset = transformation * dataset;

  if (impBounds && iteration++ % updateInterval == 0)
  {
    // Track number of data points to use for impostors calculatiom.
    size_t numPoints = 0;

    for (size_t i = begin; i < begin + batchSize; ++i)
    {
      if (lastTransformationIndices(i))
      {
        if (transformationDiffs[lastTransformationIndices[i]] *
            (2 * norm(i) + norm(impostors(k - 1, i)) +
            norm(impostors(k, i))) > distanceMat(k, i) - distanceMat(k - 1, i))
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
    constraint.Impostors(impostors, distanceMat,
        transformedDataset, labels, norm, points, numPoints);
  }
  else if (iteration++ % updateInterval == 0)
  {
    // Re-calculate impostors on transformed dataset.
    constraint.Impostors(impostors, distanceMat, transformedDataset, labels,
        norm, begin, batchSize);
  }

  gradient.zeros(transformation.n_rows, transformation.n_cols);

  MatType cij = zeros<MatType>(dataset.n_rows, dataset.n_rows);
  MatType cil = zeros<MatType>(dataset.n_rows, dataset.n_rows);

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate cost due to distance between target neighbors & data point.
      ElemType eval = distance.Evaluate(transformedDataset.col(i),
                        transformedDataset.col(targetNeighbors(j, i)));
      cost += (1 - regularization) * eval;

      // Calculate gradient due to target neighbors.
      VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      cij += diff * trans(diff);
    }

    for (int j = k - 1; j >= 0; j--)
    {
      // Bound constraints to avoid uneccesary computation.
      for (size_t l = 0, bp = k; l < bp ; l++)
      {
        // Calculate cost due to {data point, target neighbors, impostors}
        // triplets.
        ElemType eval = 0;

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
          if (iteration - 1 % updateInterval == 0)
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                 distanceMat(l, i);
          }
          else
          {
            eval = distance.Evaluate(transformedDataset.col(i),
                     transformedDataset.col(targetNeighbors(j, i))) -
                   distance.Evaluate(transformedDataset.col(i),
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
        VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
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

template<typename MatType, typename LabelsType, typename DistanceType>
inline void LMNNFunction<MatType, LabelsType, DistanceType>::Precalculate()
{
  pCij.zeros(dataset.n_rows, dataset.n_rows);

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    for (size_t j = 0; j < k ; ++j)
    {
      // Calculate gradient due to target neighbors.
      VecType diff = dataset.col(i) - dataset.col(targetNeighbors(j, i));
      pCij += diff * trans(diff);
    }
  }
}

} // namespace mlpack

#endif
