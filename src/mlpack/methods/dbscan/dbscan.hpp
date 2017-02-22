/**
 * @file dbscan.hpp
 * @author Ryan Curtin
 *
 * An implementation of the DBSCAN clustering method, which is flexible enough
 * to support other algorithms for finding nearest neighbors.
 */
#ifndef MLPACK_METHODS_DBSCAN_DBSCAN_HPP
#define MLPACK_METHODS_DBSCAN_DBSCAN_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/range_search/range_search.hpp>
#include "random_point_selection.hpp"
#include <boost/dynamic_bitset.hpp>

namespace mlpack {
namespace dbscan {

template<typename RangeSearchType = range::RangeSearch<>,
         typename PointSelectionPolicy = RandomPointSelection>
class DBSCAN
{
 public:
  /**
   * Construct the DBSCAN object with the given parameters.
   *
   * @param epsilon Size of range query.
   * @param minPoints Minimum number of points for each cluster.
   */
  DBSCAN(const double epsilon,
         const size_t minPoints,
         RangeSearchType rangeSearch = RangeSearchType(),
         PointSelectionPolicy pointSelector = PointSelectionPolicy());

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters 
   * and also the centroid of each cluster.
   *
   * @param MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param centroids Matrix in which centroids are stored.
   */
  template<typename MatType>
  size_t Cluster(const MatType& data,
                 arma::mat& centroids);

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters 
   * and also the list of cluster assignments.
   *
   * @param MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments.
   */
  template<typename MatType>
  size_t Cluster(const MatType& data,
                 arma::Row<size_t>& assignments);

  /**
   * Performs DBSCAN clustering on the data, returning number of clusters, 
   * the centroid of each cluster and also the list of cluster assignments.
   * !If assignments[i] == assignments.n_elem - 1, then the point is considered
   * !"noise".
   *
   * @param MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param assignments Vector to store cluster assignments.
   * @param centroids Matrix in which centroids are stored.
   */
  template<typename MatType>
  size_t Cluster(const MatType& data,
                 arma::Row<size_t>& assignments,
                 arma::mat& centroids);

 private:
  // Maximum distance between two points to be part of same cluster. 
  double epsilon;

  // Minimum number of points to be in the epsilon-neighborhood(including
  // itself) for the point to be a core-point.
  size_t minPoints;

  // Instantiated range search policy.
  RangeSearchType rangeSearch;

  // Instantiated point selection policy.
  PointSelectionPolicy pointSelector;

  /**
   * This function processes the point at index. It  marks the point 
   * as visited, checks if the given point is core or non-core.
   * If its a core point, it expands the cluster else returns.
   * 
   *
   * @param MatType Type of matrix (arma::mat or arma::sp_mat).
   * @param data Dataset to cluster.
   * @param unvisited Remembers if a point has been visited.
   * @param index Index of point to be visited now.
   * @param assignments Vector to store cluster assignments.
   * @param currentCluster Index of cluster which will be  
   *                       assigned to points in current cluster.
   * @param neighbor Matrix containing list of neighbors for each point
   *                 which fall in its epsilon-neighborhood.
   * @param distances Matrix containing list of distances for each point
   *                  which fall in its epsilon-neighborhood.  
   * @param topLevel If true, then current point is the first point in  
   *                 the current cluster, helps in detecting noise.
   */
  template<typename MatType>
  size_t ProcessPoint(const MatType& data,
                      boost::dynamic_bitset<>& unvisited,
                      const size_t index,
                      arma::Row<size_t>& assignments,
                      const size_t currentCluster,
                      const std::vector<std::vector<size_t>>& neighbors,
                      const std::vector<std::vector<double>>& distances,
                      const bool topLevel = true);
};

} // namespace dbscan
} // namespace mlpack

// Include implementation.
#include "dbscan_impl.hpp"

#endif
