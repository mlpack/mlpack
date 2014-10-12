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

  //! Get the number of base cases that have been performed.
  size_t BaseCases() const { return baseCases; }
  //! Modify the number of base cases that have been performed.
  size_t& BaseCases() { return baseCases; }

  //! Get the number of scores that have been performed.
  size_t Scores() const { return scores; }
  //! Modify the number of scores that have been performed.
  size_t& Scores() { return scores; }

  //! Convenience typedef.
  typedef neighbor::NeighborSearchTraversalInfo<TreeType> TraversalInfoType;

  //! Get the traversal info.
  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  //! Modify the traversal info.
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

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

  //! The number of base cases that have been performed.
  size_t baseCases;
  //! The number of scores that have been performed.
  size_t scores;

  TraversalInfoType traversalInfo;

  arma::uvec spareBlacklist;
};

}; // namespace kmeans
}; // namespace mlpack

// Include implementation.
#include "pelleg_moore_kmeans_rules_impl.hpp"

#endif
