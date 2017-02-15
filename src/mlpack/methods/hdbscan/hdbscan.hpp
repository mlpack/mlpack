/**
 * @file hdbscan.hpp
 * @author Ryan Curtin
 *
 * An implementation of the HDBSCAN clustering method.
 */
#ifndef __MLPACK_METHODS_HDBSCAN_HDBSCAN_HPP
#define __MLPACK_METHODS_HDBSCAN_HDBSCAN_HPP

#include <mlpack/core.hpp>
#include <boost/dynamic_bitset.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <mlpack/methods/hdbscan/mutualReachabilityMetric.hpp>

namespace mlpack {
namespace hdbscan {

template<typename NeighborSearch = neighbor::NeighborSearch<neighbor::NearestNeighborSort , metric::EuclideanDistance>,
         typename MetricType = metric::HdbscanDistance>
class HDBSCAN
{

  
 public:
  /**
   * Construct the HDBSCAN object with the given parameters.
   *
   * @param minPoints Minimum number of points for each cluster.
   */
  HDBSCAN(const size_t minPoints,
         NeighborSearch neighborSearch = NeighborSearch());

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

 private:
  // The parameter to help compute core distance.
  size_t minPoints;

  //Minimum number of points in the cluster.
  size_t minClusterSize;

  //Instantiated neighbor search.
  NeighborSearch neighborSearch;

  //Metric defined
  MetricType metric;
};
} // namespace hdbscan
} // namespace mlpack

// Include implementation.
#include "hdbscan_impl.hpp"

#endif
