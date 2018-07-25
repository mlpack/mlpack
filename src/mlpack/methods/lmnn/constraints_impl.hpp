/**
 * @file constraints_impl.h
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
namespace lmnn {

template<typename MetricType>
Constraints<MetricType>::Constraints(
    const arma::mat& /* dataset */,
    const arma::Row<size_t>& labels,
    const size_t k) :
    k(k),
    precalculated(false)
{
  // Ensure a valid k is passed.
  size_t minCount = arma::min(arma::histc(labels, arma::unique(labels)));

  if (minCount < k)
  {
    Log::Fatal << "Constraints::Constraints(): One of the class contains only "
        << minCount << " instances, but value of k is " << k << "  "
        << "(k should be < " << minCount << ")!" << std::endl;
  }
}

// We assume the dataset held in the tree has already been stretched.
template<typename TreeType>
inline void UpdateTree(TreeType& node)
{
  double minWidth = DBL_MAX;
  arma::Col<typename TreeType::ElemType> mins(node.Dataset().n_rows);
  arma::Col<typename TreeType::ElemType> maxs(node.Dataset().n_rows);
  for (size_t d = 0; d < node.Bound().Dim(); ++d)
  {
    mins[d] = node.Dataset().col(node.Descendant(0))[d];
    maxs[d] = node.Dataset().col(node.Descendant(0))[d];
  }

  for (size_t desc = 1; desc < node.NumDescendants(); ++desc)
  {
    for (size_t d = 0; d < node.Bound().Dim(); ++d)
    {
      const double value = node.Dataset().col(node.Descendant(desc))[d];
      if (value < mins[d])
        mins[d] = value;
      if (value > maxs[d])
        maxs[d] = value;
    }
  }

  // Update each of the HRectBound dimensions...
  for (size_t d = 0; d < node.Bound().Dim(); ++d)
  {
    // Basically we had a previous bound and we want to "stretch" it for each
    // dimension...
    node.Bound()[d] = math::RangeType<double>(mins[d], maxs[d]);

    // Calculate minWidth.
    double width = node.Bound()[d].Width();
    if (width < minWidth)
      minWidth = width;
  }

  node.Bound().MinWidth() = minWidth;

  // Technically this is loose but it is what the BinarySpaceTree already does.
  node.FurthestDescendantDistance() = 0.5 * node.Bound().Diameter();

  // Recurse into the children.
  if (node.NumChildren() > 0)
  {
    UpdateTree(node.Child(0));
    UpdateTree(node.Child(1));

    // Recompute the parent distance for the left and right child.
    arma::vec center, leftCenter, rightCenter;
    node.Center(center);
    node.Child(0).Center(leftCenter);
    node.Child(1).Center(rightCenter);

    const double leftParentDistance = node.Metric().Evaluate(center,
        leftCenter);
    const double rightParentDistance = node.Metric().Evaluate(center,
        rightCenter);

    node.Child(0).ParentDistance() = leftParentDistance;
    node.Child(1).ParentDistance() = rightParentDistance;
  }
}

// Calculates k similar labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                                              const arma::mat& dataset,
                                              const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame[i]));
    knn.Search(k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexSame[i].at(neighbors(j));

    // Store target neihbors.
    outputMatrix.cols(indexSame[i]) = neighbors;
  }
}

// Calculates k similar labeled nearest neighbors  on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::TargetNeighbors(arma::Mat<size_t>& outputMatrix,
                                              const arma::mat& dataset,
                                              const arma::Row<size_t>& labels,
                                              const size_t begin,
                                              const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate Target Neighbors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with same class points as both reference
    // set and query set.
    knn.Train(dataset.cols(indexSame[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexSame[i].at(neighbors(j));

    // Store target neighbors.
    outputMatrix.cols(begin + subIndexSame) = neighbors;
  }
}

// Calculates k differently labeled nearest neighbors.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Perform KNN search with differently labeled points as reference
    // set and  same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(dataset.cols(indexSame[i]), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(indexSame[i]) =  neighbors;
  }
}

// Calculates k differently labeled nearest neighbors. The function
// writes back calculated neighbors & distances to passed matrices.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const arma::mat& transformation)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    if (knnObjects.size() == uniqueLabels.n_elem)
    {
      knnObjects[i].ReferenceTree().Dataset() =
        transformation * origDatasets[i];
      UpdateTree(knnObjects[i].ReferenceTree());

      knnObjects[i].Search(dataset.cols(indexSame[i]), k, neighbors, distances);
    }
    else
    {
      // KNN instance.
      KNN knn;
      knn.Train(dataset.cols(indexDiff[i]));
      knn.Search(dataset.cols(indexSame[i]), k, neighbors, distances);

      // Cache KNN object.
      knnObjects.push_back(knn);
      origDatasets.push_back(knn.ReferenceTree().Dataset());
    }

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(indexSame[i]) =  neighbors;
    outputDistance.cols(indexSame[i]) =  distances;
  }
}

// Calculates k differently labeled nearest neighbors on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputMatrix,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const size_t begin,
                                        const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate impostors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputMatrix.cols(begin + subIndexSame) =  neighbors;
  }
}

// Calculates k differently labeled nearest neighbors & distances on a
// batch of data points.
template<typename MetricType>
void Constraints<MetricType>::Impostors(arma::Mat<size_t>& outputNeighbors,
                                        arma::mat& outputDistance,
                                        const arma::mat& dataset,
                                        const arma::Row<size_t>& labels,
                                        const size_t begin,
                                        const size_t batchSize)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  arma::mat subDataset = dataset.cols(begin, begin + batchSize - 1);
  arma::Row<size_t> sublabels = labels.cols(begin, begin + batchSize - 1);

  // KNN instance.
  KNN knn;

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  // Vectors to store indices.
  arma::uvec subIndexSame;

  for (size_t i = 0; i < uniqueLabels.n_cols; i++)
  {
    // Calculate impostors.
    subIndexSame = arma::find(sublabels == uniqueLabels[i]);

    // Perform KNN search with differently labeled points as reference
    // set and same class points as query set.
    knn.Train(dataset.cols(indexDiff[i]));
    knn.Search(subDataset.cols(subIndexSame), k, neighbors, distances);

    // Re-map neighbors to their index.
    for (size_t j = 0; j < neighbors.n_elem; j++)
      neighbors(j) = indexDiff[i].at(neighbors(j));

    // Store impostors.
    outputNeighbors.cols(begin + subIndexSame) =  neighbors;
    outputDistance.cols(begin + subIndexSame) =  distances;
  }
}

// Generates {data point, target neighbors, impostors} triplets using
// TargetNeighbors() and Impostors().
template<typename MetricType>
void Constraints<MetricType>::Triplets(arma::Mat<size_t>& outputMatrix,
                                       const arma::mat& dataset,
                                       const arma::Row<size_t>& labels)
{
  // Perform pre-calculation. If neccesary.
  Precalculate(labels);

  size_t N = dataset.n_cols;

  arma::Mat<size_t> impostors;
  Impostors(impostors, dataset);

  arma::Mat<size_t> targetNeighbors;
  TargetNeighbors(targetNeighbors, dataset);

  outputMatrix = arma::Mat<size_t>(3, k * k * N , arma::fill::zeros);

  for (size_t i = 0, r = 0; i < N; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      for (size_t l = 0; l < k; l++, r++)
      {
        // Generate triplets.
        outputMatrix(0, r) = i;
        outputMatrix(1, r) = targetNeighbors(j, i);
        outputMatrix(2, r) = impostors(l, i);
      }
    }
  }
}

template<typename MetricType>
inline void Constraints<MetricType>::Precalculate(
                                         const arma::Row<size_t>& labels)
{
  // Make sure the calculation is necessary.
  if (precalculated)
    return;

  uniqueLabels = arma::unique(labels);

  indexSame.resize(uniqueLabels.n_elem);
  indexDiff.resize(uniqueLabels.n_elem);

  for (size_t i = 0; i < uniqueLabels.n_elem; i++)
  {
    // Store same and diff indices.
    indexSame[i] = arma::find(labels == uniqueLabels[i]);
    indexDiff[i] = arma::find(labels != uniqueLabels[i]);
  }

  precalculated = true;
}

} // namespace lmnn
} // namespace mlpack

#endif
