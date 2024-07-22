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
#include <omp.h>

namespace mlpack {

/**
 * Construct the DBSCAN object with the given parameters.
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
DBSCAN<RangeSearchType, PointSelectionPolicy>::DBSCAN(
    const ElemType epsilon,
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
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
    const MatType& data,
    MatType& centroids)
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
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
    const MatType& data,
    arma::Row<size_t>& assignments,
    MatType& centroids)
{
  const size_t numClusters = Cluster(data, assignments);

  // Now calculate the centroids.
  centroids.zeros(data.n_rows, numClusters);

  // Calculate number of points in each cluster.
  arma::Row<size_t> counts;
  counts.zeros(numClusters);

  const size_t numThreads = omp_get_max_threads();
  std::vector<arma::Row<size_t>> localCounts(numThreads, arma::Row<size_t>(numClusters, arma::fill::zeros));
  std::vector<MatType> localCentroids(numThreads, MatType(data.n_rows, numClusters, arma::fill::zeros));

  #pragma omp parallel
  {
    const size_t threadId = omp_get_thread_num();

    #pragma omp for nowait
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      if (assignments[i] != SIZE_MAX)
      {
        localCentroids[threadId].col(assignments[i]) += data.col(i);
        ++localCounts[threadId][assignments[i]];
      }
    }
  }

  // Combine results from all threads
  for (size_t i = 0; i < numThreads; ++i)
  {
    centroids += localCentroids[i];
    counts += localCounts[i];
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
  #pragma omp parallel for
  for (size_t i = 0; i < data.n_cols; ++i)
    assignments[i] = uf.Find(i);

  // Get a count of all clusters.
  const size_t numClusters = max(assignments) + 1;
  arma::Col<size_t> counts(numClusters, arma::fill::zeros);

  const size_t numThreads = omp_get_max_threads();
  std::vector<arma::Col<size_t>> localCounts(numThreads, arma::Col<size_t>(numClusters, arma::fill::zeros));

  #pragma omp parallel
  {
    const size_t threadId = omp_get_thread_num();

    #pragma omp for nowait
    for (size_t i = 0; i < assignments.n_elem; ++i)
      ++localCounts[threadId][assignments[i]];
  }

  // Combine results from all threads
  for (size_t i = 0; i < numThreads; ++i)
    counts += localCounts[i];


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
  #pragma omp parallel for
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
void DBSCAN<RangeSearchType, PointSelectionPolicy>::PointwiseCluster(
    const MatType& data,
    UnionFind& uf)
{
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<ElemType>> distances;

  std::vector<bool> visited(data.n_cols, false);
  std::vector<bool> nonCorePoints(data.n_cols, false);

  #pragma omp parallel for schedule(dynamic) firstprivate(neighbors, distances)
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (i % 10000 == 0 && i > 0)
      Log::Info << "DBSCAN clustering on point " << i << "..." << std::endl;

    // Get the next index.
    const size_t index = pointSelector.Select(i, data);
    visited[index] = true;

    // Do the range search for only this point.
    rangeSearch.Search(data.col(index), RangeType<ElemType>(ElemType(0.0), epsilon),
        neighbors, distances);

    if (neighbors[0].size() >= minPoints)
    {
      for (size_t j = 0; j < neighbors[0].size(); ++j)
      {
        if (uf.Find(neighbors[0][j]) == neighbors[0][j])
        {
          #pragma omp critical
          uf.Union(index, neighbors[0][j]);
        }
        else if (!nonCorePoints[neighbors[0][j]] && visited[neighbors[0][j]])
        {
          #pragma omp critical
          uf.Union(index, neighbors[0][j]);
        }
      }
    }
    else
    {
      nonCorePoints[index] = true;
    }
  }
}

/**
 * Performs DBSCAN clustering on the data, returning number of clusters
 * and also the list of cluster assignments.  This can perform search in batch,
 * naive search).
 */
template<typename RangeSearchType, typename PointSelectionPolicy>
void DBSCAN<RangeSearchType, PointSelectionPolicy>::BatchCluster(
    const MatType& data,
    UnionFind& uf)
{
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<ElemType>> distances;
  Log::Info << "Performing range search." << std::endl;
  rangeSearch.Train(data);
  rangeSearch.Search(RangeType<ElemType>(ElemType(0.0), epsilon), neighbors,
      distances);
  Log::Info << "Range search complete." << std::endl;

  #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const size_t index = pointSelector.Select(i, data);
    if (neighbors[index].size() >= minPoints - 1)
    {
      for (size_t j = 0; j < neighbors[index].size(); ++j)
      {
        if (uf.Find(neighbors[index][j]) == neighbors[index][j])
        {
          #pragma omp critical
          uf.Union(index, neighbors[index][j]);
        }
        else if (neighbors[neighbors[index][j]].size() >= (minPoints - 1))
        {
          #pragma omp critical
          uf.Union(index, neighbors[index][j]);
        }
      }
    }
  }
}

} // namespace mlpack

#endif
