/**
 * @file methods/dbscan/dbscan_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DBSCAN.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DBSCAN_DBSCAN_IMPL_HPP
#define MLPACK_METHODS_DBSCAN_DBSCAN_IMPL_HPP

#include "dbscan.hpp"

namespace mlpack {

/**
 * Construct the DBSCAN object with the given parameters.
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
DBSCAN<RangeSearchType, PointSelectionPolicy>::DBSCAN(
    const double epsilon,
    const size_t minPoints,
    const bool batchMode,
    RangeSearchType rangeSearch,
    PointSelectionPolicy pointSelector) :
    epsilon(epsilon),
    minPoints(minPoints),
    batchMode(batchMode),
    rangeSearch(rangeSearch),
    pointSelector(pointSelector)
{
  // Nothing to do.
}

/**
 * Performs DBSCAN clustering on the data, returning number of clusters
 * and also the centroid of each cluster.
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
    const MatType& data,
    arma::mat& centroids)
{
  // These assignments will be thrown away, but there is no way to avoid
  // calculating them.
  arma::Row<size_t> assignments(data.n_cols);
  assignments.fill(SIZE_MAX);

  return Cluster(data, assignments, centroids);
}

/**
 * Performs DBSCAN clustering on the data, returning number of clusters,
 * the centroid of each cluster and also the list of cluster assignments.
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
    const MatType& data,
    arma::Row<size_t>& assignments,
    arma::mat& centroids)
{
  const size_t numClusters = Cluster(data, assignments);

  // Now calculate the centroids.
  centroids.zeros(data.n_rows, numClusters);

  // Calculate number of points in each cluster.
  arma::Row<size_t> counts;
  counts.zeros(numClusters);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (assignments[i] != SIZE_MAX)
    {
      centroids.col(assignments[i]) += data.col(i);
      ++counts[assignments[i]];
    }
  }

  // We should be guaranteed that the number of clusters is always greater than
  // zero.
  for (size_t i = 0; i < numClusters; ++i)
    centroids.col(i) /= counts[i];

  return numClusters;
}

/**
 * Performs DBSCAN clustering on the data, returning the number of clusters and
 * also the list of cluster assignments.
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
    const MatType& data,
    arma::Row<size_t>& assignments)
{
  // Initialize the UnionFind object.
  UnionFind uf(data.n_cols);
  rangeSearch.Train(data);

  if (batchMode)
    BatchCluster(data, uf);
  else
    PointwiseCluster(data, uf);

  // Now set assignments.
  assignments.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    assignments[i] = uf.Find(i);

  // Get a count of all clusters.
  const size_t numClusters = arma::max(assignments) + 1;
  arma::Col<size_t> counts(numClusters, arma::fill::zeros);
  for (size_t i = 0; i < assignments.n_elem; ++i)
    counts[assignments[i]]++;

  // Now assign clusters to new indices.
  size_t currentCluster = 0;
  arma::Col<size_t> newAssignments(numClusters);
  for (size_t i = 0; i < counts.n_elem; ++i)
  {
    if (counts[i] >= minPoints)
      newAssignments[i] = currentCluster++;
    else
      newAssignments[i] = SIZE_MAX;
  }

  // Now reassign.
  for (size_t i = 0; i < assignments.n_elem; ++i)
    assignments[i] = newAssignments[assignments[i]];

  Log::Info << currentCluster << " clusters found." << std::endl;

  return currentCluster;
}

/**
 * Performs DBSCAN clustering on the data, returning the number of clusters and
 * also the list of cluster assignments.  This searches each point iteratively,
 * and can save on RAM usage.  It may be slower than the batch search with a
 * dual-tree algorithm.
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
void DBSCAN<RangeSearchType, PointSelectionPolicy>::PointwiseCluster(
    const MatType& data,
    UnionFind& uf)
{
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<double>> distances;

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (i % 10000 == 0 && i > 0)
      Log::Info << "DBSCAN clustering on point " << i << "..." << std::endl;

    // Do the range search for only this point.
    rangeSearch.Search(data.col(i), Range(0.0, epsilon), neighbors,
        distances);

    // Union to all neighbors.
    for (size_t j = 0; j < neighbors[0].size(); ++j)
      uf.Union(i, neighbors[0][j]);
  }
}

/**
 * Performs DBSCAN clustering on the data, returning number of clusters
 * and also the list of cluster assignments.  This can perform search in batch,
 * naive search).
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
void DBSCAN<RangeSearchType, PointSelectionPolicy>::BatchCluster(
    const MatType& data,
    UnionFind& uf)
{
  // For each point, find the points in epsilon-nighborhood and their distances.
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<double>> distances;
  Log::Info << "Performing range search." << std::endl;
  rangeSearch.Train(data);
  rangeSearch.Search(data, Range(0.0, epsilon), neighbors, distances);
  Log::Info << "Range search complete." << std::endl;

  // Now loop over all points.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Get the next index.
    const size_t index = pointSelector.Select(i, data);
    for (size_t j = 0; j < neighbors[index].size(); ++j)
      uf.Union(index, neighbors[index][j]);
  }
}

} // namespace mlpack

#endif
