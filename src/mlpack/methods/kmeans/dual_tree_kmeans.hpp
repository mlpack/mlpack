/**
 * @file dual_tree_kmeans.hpp
 * @author Ryan Curtin
 *
 * A dual-tree algorithm for a single k-means iteration.
 */
#ifndef __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_HPP
#define __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_HPP

#include "dual_tree_kmeans_statistic.hpp"

namespace mlpack {
namespace kmeans {

template<
    typename MetricType,
    typename MatType,
    typename TreeType = tree::BinarySpaceTree<bound::HRectBound<2>,
        DualTreeKMeansStatistic>
>
class DualTreeKMeans
{
 public:
  DualTreeKMeans(const MatType& dataset, MetricType& metric);

  ~DualTreeKMeans();

  double Iterate(const arma::mat& centroids,
                 arma::mat& newCentroids,
                 arma::Col<size_t>& counts);

  //! Return the number of distance calculations.
  size_t DistanceCalculations() const { return distanceCalculations; }
  //! Modify the number of distance calculations.
  size_t& DistanceCalculations() { return distanceCalculations; }

 private:
  //! The original dataset reference.
  const MatType& datasetOrig;
  //! The dataset we are using.
  const MatType& dataset;
  //! A copy of the dataset, if necessary.
  MatType datasetCopy;
  //! The metric.
  MetricType metric;

  //! The tree built on the points.
  TreeType* tree;

  arma::vec clusterDistances;
  arma::Col<size_t> assignments;
  arma::vec distances;
  arma::Col<size_t> distanceIteration;

  //! The current iteration.
  size_t iteration;

  //! Track distance calculations.
  size_t distanceCalculations;

  void ClusterTreeUpdate(TreeType* node,
                         const arma::mat& distances);

  void UpdateOwner(TreeType* node,
                   const size_t clusters,
                   const arma::Col<size_t>& assignments) const;

  void TreeUpdate(TreeType* node,
                  const size_t clusters,
                  const arma::vec& clusterDistances,
                  const arma::Col<size_t>& assignments,
                  const arma::mat& oldCentroids,
                  const arma::mat& dataset,
                  const std::vector<size_t>& oldFromNew,
                  size_t& hamerlyPruned);
};

template<typename MetricType, typename MatType>
using DefaultDualTreeKMeans = DualTreeKMeans<MetricType, MatType>;

} // namespace kmeans
} // namespace mlpack

// Include implementation.
#include "dual_tree_kmeans_impl.hpp"

#endif
