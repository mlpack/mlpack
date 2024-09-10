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
  const size_t numClusters = max(assignments) + 1;
  arma::Col<size_t> counts(numClusters);
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
void DBSCAN<RangeSearchType, PointSelectionPolicy>::PointwiseCluster(
    const MatType& data,
    UnionFind& uf)
{
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<ElemType>> distances;

  // Note that the strategy here is somewhat different from the original DBSCAN
  // paper.  The original DBSCAN paper grows each cluster individually to its
  // fullest extent; here, we use a UnionFind structure to grow each point into
  // a local cluster (if it has enough points), and we combine with other local
  // clusters.  The end result is the same.
  //
  // Define points as being either core points, or non-core points.  Core points
  // have more than `minPoints` neighbors.  Non-core points are included into
  // the first core point cluster that encounters them; if they are not included
  // by anything, they are labeled as noise.
  //
  // We maintain a list of non-core points so that we can handle that logic
  // correctly.

  std::vector<bool> visited(data.n_cols, false);
  std::vector<bool> nonCorePoints(data.n_cols, false);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (i % 10000 == 0 && i > 0)
      Log::Info << "DBSCAN clustering on point " << i << "..." << std::endl;

    // Get the next index.
    const size_t index = pointSelector.Select(i, data);
    visited[index] = true;

    // Do the range search for only this point.
    rangeSearch.Search(data.col(index),
        RangeType<ElemType>(ElemType(0.0), epsilon), neighbors, distances);

    // Union to all neighbors if the point is not noise.
    //
    // If the point is noise, we leave its label as undefined (i.e. we do no
    // unioning).
    if (neighbors[0].size() >= minPoints)
    {
      for (size_t j = 0; j < neighbors[0].size(); ++j)
      {
        // Union to all neighbors that either do not have a label, or are core
        // points of other clusters.  (When we union to another core point, we
        // are merging clusters.)
        if (uf.Find(neighbors[0][j]) == neighbors[0][j])
        {
          // This unions unlabeled points.
          uf.Union(index, neighbors[0][j]);
        }
        else if (!nonCorePoints[neighbors[0][j]] && visited[neighbors[0][j]])
        {
          // This unions core points of other clusters.  Note that we only union
          // with other clusters that have already been visited---this is
          // because we do not know whether unvisited points are core or
          // non-core points.  (If an unvisited point is a core point, it'll
          // merge with us later.)
          uf.Union(index, neighbors[0][j]);
        }
      }
    }
    else
    {
      // This is not a core point---it does not have enough neighbors.
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
  // For each point, find the points in epsilon-neighborhood and their
  // distances.
  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<ElemType>> distances;
  Log::Info << "Performing range search." << std::endl;
  rangeSearch.Train(data);
  rangeSearch.Search(RangeType<ElemType>(ElemType(0.0), epsilon), neighbors,
      distances);
  Log::Info << "Range search complete." << std::endl;

  // See the description of the algorithm in `PointwiseCluster()`.  The strategy
  // is the same here, but we have cached all range search results already.
  // That means we already have computed whether each point is or is not a core
  // point, just based on the size of its neighbors; so we don't need an
  // auxiliary std::vector<bool> for that.

  // Now loop over all points.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Get the next index.
    const size_t index = pointSelector.Select(i, data);
    // Monochromatic dual-tree range search does not return the point as its own
    // neighbor, so we are looking for `minPoints - 1` instead.
    if (neighbors[index].size() >= minPoints - 1)
    {
      for (size_t j = 0; j < neighbors[index].size(); ++j)
      {
        if (uf.Find(neighbors[index][j]) == neighbors[index][j])
        {
          // This unions unlabeled points.
          uf.Union(index, neighbors[index][j]);
        }
        else if (neighbors[neighbors[index][j]].size() >= (minPoints - 1))
        {
          // This unions core points of other clusters.
          uf.Union(index, neighbors[index][j]);
        }
      }
    }
  }
}

} // namespace mlpack

#endif
