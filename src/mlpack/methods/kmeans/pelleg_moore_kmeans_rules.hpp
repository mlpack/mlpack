/**
 * @file kmeans_rules.hpp
 * @author Ryan Curtin
 *
 * Defines the pruning rules and base cases rules necessary to perform
 * single-tree k-means clustering using the Pelleg-Moore fast k-means algorithm,
 * which has been shoehorned to fit into the mlpack tree abstractions.
 */
#ifndef __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_HPP
#define __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_HPP

#include <mlpack/methods/neighbor_search/ns_traversal_info.hpp>

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
class PellegMooreKMeansRules
{
 public:
  PellegMooreKMeansRules(const typename TreeType::Mat& dataset,
                         const arma::mat& centroids,
                         arma::mat& newCentroids,
                         arma::Col<size_t>& counts,
                         MetricType& metric);

  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  double Score(const size_t queryIndex, TreeType& referenceNode);

  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore);

  //! Get the number of distance calculations that have been performed.
  size_t DistanceCalculations() const { return distanceCalculations; }
  //! Modify the number of distance calculations that have been performed.
  size_t& DistanceCalculations() { return distanceCalculations; }

 private:
  //! The dataset.
  const typename TreeType::Mat& dataset;
  //! The clusters.
  const arma::mat& centroids;
  //! The new centroids.
  arma::mat& newCentroids;
  //! The counts of points in each cluster.
  arma::Col<size_t>& counts;
  //! Instantiated metric.
  MetricType& metric;

  //! The number of O(d) distance calculations that have been performed.
  size_t distanceCalculations;

  //! Spare blacklist; I think it's only used by the root node.
  arma::uvec spareBlacklist;
};

}; // namespace kmeans
}; // namespace mlpack

// Include implementation.
#include "pelleg_moore_kmeans_rules_impl.hpp"

#endif
