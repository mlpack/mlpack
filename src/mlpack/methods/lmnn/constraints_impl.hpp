/**
 * @file methods/lmnn/constraints_impl.hpp
 * @author Manish Kumar
 *
 * Implementation of the Constraints class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP
#define MLPACK_METHODS_LMNN_CONSTRAINTS_IMPL_HPP

// In case it hasn't been included already.
#include "constraints.hpp"

namespace mlpack {

template<typename MatType, typename LabelsType, typename DistanceType>
Constraints<MatType, LabelsType, DistanceType>::Constraints(
    const MatType& /* dataset */,
    const LabelsType& labels,
    const size_t k) :
    k(k),
    precalculated(false)
{
  // Ensure a valid k is passed.
  size_t minCount = min(arma::histc(labels, arma::unique(labels)));

  if (minCount < k + 1)
  {
    Log::Fatal << "Constraints::Constraints(): One of the class contains only "
        << minCount << " instances, but value of k is " << k << "  "
        << "(k should be < " << minCount << ")!" << std::endl;
  }
}

template<typename MatType, typename LabelsType, typename DistanceType>
inline void Constraints<MatType, LabelsType, DistanceType>::ReorderResults(
    const MatType& distances,
    UMatType& neighbors,
    const VecType& norms)
{
  // Shortcut...
  if (neighbors.n_rows == 1)
    return;

  // Just a simple loop over the results---we want to make sure that the
  // largest-norm point with identical distance has the last location.
  for (size_t i = 0; i < neighbors.n_cols; ++i)
  {
    for (size_t start = 0; start < neighbors.n_rows - 1; start++)
    {
      size_t end = start + 1;
      while (distances(start, i) == distances(end, i) &&
          end < neighbors.n_rows)
      {
        end++;
        if (end == neighbors.n_rows)
          break;
      }

      if (start != end)
      {
        // We must sort these elements by norm.
        UVecType indices = neighbors.col(i).subvec(start, end - 1);
        UVecType order = arma::sort_index(norms.elem(indices));
        neighbors.col(i).subvec(start, end - 1) = indices.elem(order);
      }
    }
  }
}

// Calculates k similar labeled nearest neighbors.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::TargetNeighbors(
    UMatType& outputMatrix,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame[i]));
    knn.Search(k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexSame[i].at(neighbors(j));

    // Store target neihbors.
    outputMatrix.cols(indexSame[i]) = neighbors;
  }
}

// Calculates k similar labeled nearest neighbors  on a
// batch of data points.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::TargetNeighbors(
    UMatType& outputMatrix,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms,
    const size_t begin,
    const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  MatType subDataset = dataset.cols(begin, begin + batchSize - 1);
  LabelsType sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  // Vectors to store indices.
  UVecType subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Calculate Target Neighbors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexSame[i].at(neighbors(j));

    // Store target neighbors.
    outputMatrix.cols(begin + subIndexSame) = neighbors;
  }
}

// Calculates k differently labeled nearest neighbors.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::Impostors(
    UMatType& outputMatrix,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Perform KNN search with differently labeled points as reference
    // set and  same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(dataset.cols(indexSame[i]), k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(indexSame[i]) = neighbors;
  }
}

// Calculates k differently labeled nearest neighbors. The function
// writes back calculated neighbors & distances to passed matrices.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::Impostors(
    UMatType& outputNeighbors,
    MatType& outputDistance,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Perform KNN search with differently labeled points as reference
    // set and  same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(dataset.cols(indexSame[i]), k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputNeighbors.cols(indexSame[i]) = neighbors;
    outputDistance.cols(indexSame[i]) = distances;
  }
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::Impostors(
    UMatType& outputMatrix,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms,
    const size_t begin,
    const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  MatType subDataset = dataset.cols(begin, begin + batchSize - 1);
  LabelsType sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  // Vectors to store indices.
  UVecType subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Calculate impostors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(begin + subIndexSame) = neighbors;
  }
}

// Calculates k differently labeled nearest neighbors & distances on a
// batch of data points.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::Impostors(
    UMatType& outputNeighbors,
    MatType& outputDistance,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms,
    const size_t begin,
    const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  MatType subDataset = dataset.cols(begin, begin + batchSize - 1);
  LabelsType sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  // Vectors to store indices.
  UVecType subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Calculate impostors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputNeighbors.cols(begin + subIndexSame) = neighbors;
    outputDistance.cols(begin + subIndexSame) = distances;
  }
}

// Calculates k differently labeled nearest neighbors & distances over some
// data points.
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::Impostors(
    UMatType& outputNeighbors,
    MatType& outputDistance,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms,
    const UVecType& points,
    const size_t numPoints)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  UMatType neighbors;
  MatType distances;

  // Vectors to store indices.
  UVecType subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; ++i)
  {
    // Calculate impostors.
    subIndexSame = arma::find(labels.cols(points.head(numPoints)) ==
        uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(dataset.cols(points.elem(subIndexSame)),
        k, neighbors, distances);

    // Re-order neighbors on the basis of increasing norm in case
    // of ties among distances.
    ReorderResults(distances, neighbors, norms);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; ++j)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputNeighbors.cols(points.elem(subIndexSame)) = neighbors;
    outputDistance.cols(points.elem(subIndexSame)) = distances;
  }
}

// Generates {data point, target neighbors, impostors} triplets using
// TargetNeighbors() and Impostors().
template<typename MatType, typename LabelsType, typename DistanceType>
void Constraints<MatType, LabelsType, DistanceType>::Triplets(
    UMatType& outputMatrix,
    const MatType& dataset,
    const LabelsType& labels,
    const VecType& norms)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  size_t N = dataset.n_cols;

  UMatType impostors(k, dataset.n_cols);
  Impostors(impostors, dataset, labels, norms);

  UMatType targetNeighbors(k, dataset.n_cols);
  TargetNeighbors(targetNeighbors, dataset, labels, norms);

  outputMatrix = UMatType(3, k * k * N);

  #pragma omp parallel for collapse(3)
  for (size_t i = 0; i < N; ++i)
  {
    for (size_t j = 0; j < k; ++j)
    {
      for (size_t l = 0; l < k; l++)
      {
        const size_t r = i * (k * k) + j * k + l;

        // Generate triplets.
        outputMatrix(0, r) = i;
        outputMatrix(1, r) = targetNeighbors(j, i);
        outputMatrix(2, r) = impostors(l, i);
      }
    }
  }
}

template<typename MatType, typename LabelsType, typename DistanceType>
inline void Constraints<MatType, LabelsType, DistanceType>::Precalculate(
    const LabelsType& labels)
{
  // Make sure the calculation is necessary.
  if (precalculated)
    return;

  uniqueLabels = arma::unique(labels);

  indexSame.resize(uniqueLabels.n_elem);
  indexDiff.resize(uniqueLabels.n_elem);

  #pragma omp parallel for
  for (size_t i = 0; i < uniqueLabels.n_elem; ++i)
  {
    // Store same and diff indices.
    indexSame[i] = arma::find(labels == uniqueLabels[i]);
    indexDiff[i] = arma::find(labels != uniqueLabels[i]);
  }

  precalculated = true;
}

} // namespace mlpack

#endif
