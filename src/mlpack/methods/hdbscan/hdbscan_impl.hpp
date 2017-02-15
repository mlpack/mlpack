/**
 * @file hdbscan_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DBSCAN.
 */
#ifndef __MLPACK_METHODS_HDBSCAN_HDBSCAN_IMPL_HPP
#define __MLPACK_METHODS_HDBSCAN_HDBSCAN_IMPL_HPP

#include "hdbscan.hpp"

namespace mlpack {
namespace hdbscan {

/**
 * Construct the HDBSCAN object with the given parameters.
 */
template<typename NeighborSearch, typename MetricType>
HDBSCAN<NeighborSearch, MetricType>::HDBSCAN(
    const size_t minPoints,
    NeighborSearch neighborSearch) :
    minPoints(minPoints),
    neighborSearch(neighborSearch)
{
  // Nothing to do.
}

template<typename NeighborSearch, typename MetricType>
template<typename MatType>
size_t HDBSCAN<NeighborSearch,  MetricType>::Cluster(const MatType& data,
                 arma::Row<size_t>& assignments)
{
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  neighborSearch.Train(data);
  neighborSearch.Search(minPoints-1, neighbors, distances);
  
  arma::Col<double> dcore1(distances.col(distances.n_cols-1));
  
  arma::Row<double> dcore = trans(dcore1);
  
  arma::mat dataWithDcore = arma::conv_to<arma::mat>::from(data);
  
  dataWithDcore.resize(dataWithDcore.n_rows+1, dataWithDcore.n_cols);  
  
  dataWithDcore.row(dataWithDcore.n_rows-1) = dcore.row(0);
  
  emst::DualTreeBoruvka<metric::HdbscanDistance, arma::mat, tree::KDTree> dtb(dataWithDcore, false, metric);

  arma::mat results;
  dtb.ComputeMST(results);

  return dcore.n_rows;
}

} // namespace hdbscan
} // namespace mlpack

#endif
