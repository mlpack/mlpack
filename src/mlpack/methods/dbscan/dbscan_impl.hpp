/**
 * @file dbscan_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DBSCAN.
 */
#ifndef __MLPACK_METHODS_DBSCAN_DBSCAN_IMPL_HPP
#define __MLPACK_METHODS_DBSCAN_DBSCAN_IMPL_HPP

#include "dbscan.hpp"

namespace mlpack {
namespace dbscan {

template<typename RangeSearchType, typename PointSelectionPolicy>
DBSCAN<RangeSearchType, PointSelectionPolicy>::DBSCAN(
    const double epsilon,
    const size_t minPoints,
    RangeSearchType rangeSearch,
    PointSelectionPolicy pointSelector) :
    epsilon(epsilon),
    minPoints(minPoints),
    rangeSearch(rangeSearch),
    pointSelector(pointSelector)
{
  // Nothing to do.
}

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

template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::Cluster(
    const MatType& data,
    arma::Row<size_t>& assignments)
{
  assignments.set_size(data.n_cols);
  assignments.fill(SIZE_MAX);

  size_t currentCluster = 0;

  std::vector<std::vector<size_t>> neighbors;
  std::vector<std::vector<double>> distances;
  rangeSearch.Train(data);
  rangeSearch.Search(data, math::Range(0.0, epsilon), neighbors, distances);

  // Initialize to all true; false means it's been visited.
  boost::dynamic_bitset<> unvisited(data.n_cols);
  unvisited.set();
  while (unvisited.any())
  {
    const size_t nextIndex = pointSelector.Select(unvisited, data);

    // currentCluster will only be incremented if a cluster was created.
    currentCluster = ProcessPoint(data, unvisited, nextIndex, assignments,
        currentCluster, neighbors, distances);
  }

  return currentCluster;
}

template<typename RangeSearchType, typename PointSelectionPolicy>
template<typename MatType>
size_t DBSCAN<RangeSearchType, PointSelectionPolicy>::ProcessPoint(
    const MatType& data,
    boost::dynamic_bitset<>& unvisited,
    const size_t index,
    arma::Row<size_t>& assignments,
    const size_t currentCluster,
    const std::vector<std::vector<size_t>>& neighbors,
    const std::vector<std::vector<double>>& distances,
    const bool topLevel)
{
  // We've now visited this point.
  unvisited[index] = false;

  if ((neighbors[index].size() < minPoints) && topLevel)
  {
    // Mark the point as noise (leave assignments[index] unset) and return.
    unvisited[index] = false;
    return currentCluster;
  }
  else
  {
    assignments[index] = currentCluster;

    // New cluster.
    for (size_t j = 0; j < neighbors[index].size(); ++j)
    {
      // Add each point to the cluster and mark it as visited, but only if it
      // has not been visited yet.
      if (!unvisited[neighbors[index][j]])
        continue;

      assignments[neighbors[index][j]] = currentCluster;
      unvisited[neighbors[index][j]] = false;

      // Recurse into this point.
      ProcessPoint(data, unvisited, neighbors[index][j], assignments,
          currentCluster, neighbors, distances, false);
    }
    return currentCluster + 1;
  }
}

} // namespace dbscan
} // namespace mlpack

#endif
