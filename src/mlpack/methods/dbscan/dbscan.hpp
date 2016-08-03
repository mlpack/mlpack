/**
 * @file dbscan.hpp
 * @author Ryan Curtin
 *
 * An implementation of the DBSCAN clustering method, which is flexible enough
 * to support other algorithms for finding nearest neighbors.
 */
#ifndef __MLPACK_METHODS_DBSCAN_DBSCAN_HPP
#define __MLPACK_METHODS_DBSCAN_DBSCAN_HPP

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
         const size_t minPoints);

  template<typename MatType>
  size_t Cluster(const MatType& data,
                 arma::mat& centroids);

  template<typename MatType>
  size_t Cluster(const MatType& data,
                 arma::Row<size_t>& assignments);

  //! If assignments[i] == assignments.n_elem - 1, then the point is considered
  //! "noise".
  template<typename MatType>
  size_t Cluster(const MatType& data,
                 arma::Row<size_t>& assignments,
                 arma::mat& centroids);

 private:
  RangeSearchType rangeSearch;
  PointSelectionPolicy pointSelector;
  double epsilon;
  size_t minPoints;

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
