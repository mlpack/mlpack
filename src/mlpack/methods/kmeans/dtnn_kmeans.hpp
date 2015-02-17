/**
 * @file dtnn_kmeans.hpp
 * @author Ryan Curtin
 *
 * An implementation of a Lloyd iteration which uses dual-tree nearest neighbor
 * search as a black box.  The conditions under which this will perform best are
 * probably limited to the case where k is close to the number of points in the
 * dataset, and the number of iterations of the k-means algorithm will be few.
 */
#ifndef __MLPACK_METHODS_KMEANS_DTNN_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_DTNN_KMEANS_HPP

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

#include "dtnn_statistic.hpp"

namespace mlpack {
namespace kmeans {

/**
 * An algorithm for an exact Lloyd iteration which simply uses dual-tree
 * nearest-neighbor search to find the nearest centroid for each point in the
 * dataset.  The conditions under which this will perform best are probably
 * limited to the case where k is close to the number of points in the dataset,
 * and the number of iterations of the k-means algorithm will be few.
 */
template<
    typename MetricType,
    typename MatType,
    typename TreeType = tree::BinarySpaceTree<bound::HRectBound<2>,
        DTNNStatistic> >
class DTNNKMeans
{
 public:
  /**
   * Construct the DTNNKMeans object, which will construct a tree on the points.
   */
  DTNNKMeans(const MatType& dataset, MetricType& metric);

  /**
   * Delete the tree constructed by the DTNNKMeans object.
   */
  ~DTNNKMeans();

  /**
   * Run a single iteration of the dual-tree nearest neighbor algorithm for
   * k-means, updating the given centroids into the newCentroids matrix.
   *
   * @param centroids Current cluster centroids.
   * @param newCentroids New cluster centroids.
   * @param counts Current counts, to be overwritten with new counts.
   */
  double Iterate(const arma::mat& centroids,
                 arma::mat& newCentroids,
                 arma::Col<size_t>& counts);

  //! Return the number of distance calculations.
  size_t DistanceCalculations() const { return distanceCalculations; }
  //! Modify the number of distance calculations.
  size_t& DistanceCalculations() { return distanceCalculations; }

 private:
  //! The original dataset reference.
  const MatType& datasetOrig; // Maybe not necessary.
  //! The dataset we are using.
  const MatType& dataset;
  //! A copy of the dataset, if necessary.
  MatType datasetCopy;
  //! The metric.
  MetricType metric;

  //! The tree built on the points.
  TreeType* tree;

  //! Track distance calculations.
  size_t distanceCalculations;
  //! Track iteration number.
  size_t iteration;

  //! Upper bounds on nearest centroid.
  arma::vec upperBounds;
  //! Lower bounds on second closest cluster distance for each point.
  arma::vec lowerBounds;
  //! Indicator of whether or not the point is pruned.
  std::vector<bool> prunedPoints;

  arma::Col<size_t> assignments;

  std::vector<bool> visited; // Was the point visited this iteration?

  arma::mat lastIterationCentroids; // For sanity checks.

  //! Update the bounds in the tree before the next iteration.
  void UpdateTree(TreeType& node,
                  arma::vec& clusterDistances,
                  std::vector<size_t>& oldFromNewCentroids,
                  arma::mat& newCentroids);

  //! Extract the centroids of the clusters.
  void ExtractCentroids(TreeType& node,
                        arma::mat& newCentroids,
                        arma::Col<size_t>& newCounts,
                        std::vector<size_t>& oldFromNewCentroids,
                        arma::mat& centroids);
};

//! A template typedef for the DTNNKMeans algorithm with the default tree type
//! (a kd-tree).
template<typename MetricType, typename MatType>
using DefaultDTNNKMeans = DTNNKMeans<MetricType, MatType>;

//! A template typedef for the DTNNKMeans algorithm with the cover tree type.
template<typename MetricType, typename MatType>
using CoverTreeDTNNKMeans = DTNNKMeans<MetricType, MatType,
    tree::CoverTree<metric::EuclideanDistance, tree::FirstPointIsRoot,
    DTNNStatistic> >;

} // namespace kmeans
} // namespace mlpack

#include "dtnn_kmeans_impl.hpp"

#endif
