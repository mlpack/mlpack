/**
 * @file dual_tree_kmeans_rules.hpp
 * @author Ryan Curtin
 *
 * A set of tree traversal rules for dual-tree k-means clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_HPP
#define __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_HPP

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
class DualTreeKMeansRules
{
 public:
  DualTreeKMeansRules(const typename TreeType::Mat& dataset,
                      const arma::mat& centroids,
                      arma::mat& newCentroids,
                      arma::Col<size_t>& counts,
                      const std::vector<size_t>& mappings,
                      const size_t iteration,
                      const arma::vec& clusterDistances,
                      arma::vec& distances,
                      arma::Col<size_t>& assignments,
                      arma::Col<size_t>& visited,
                      arma::Col<size_t>& distanceIteration,
                      arma::vec& hamerlyBounds,
                      const arma::mat& interclusterDistances,
                      MetricType& metric);

  double BaseCase(const size_t queryIndex, const size_t referenceIndex);

  double Score(const size_t queryIndex, TreeType& referenceNode);

  double Score(TreeType& queryNode, TreeType& referenceNode);

  double Rescore(const size_t queryIndex,
                 TreeType& referenceNode,
                 const double oldScore) const;

  double Rescore(TreeType& queryNode,
                 TreeType& referenceNode,
                 const double oldScore) const;

  size_t DistanceCalculations() const { return distanceCalculations; }
  size_t& DistanceCalculations() { return distanceCalculations; }

  typedef neighbor::NeighborSearchTraversalInfo<TreeType> TraversalInfoType;

  const TraversalInfoType& TraversalInfo() const { return traversalInfo; }
  TraversalInfoType& TraversalInfo() { return traversalInfo; }

 private:
  const typename TreeType::Mat& dataset;
  const arma::mat& centroids;
  arma::mat& newCentroids;
  arma::Col<size_t>& counts;
  const std::vector<size_t>& mappings;
  const size_t iteration;
  const arma::vec& clusterDistances;
  arma::vec& distances;
  arma::Col<size_t>& assignments;
  arma::Col<size_t>& visited;
  arma::Col<size_t>& distanceIteration;
  arma::vec& hamerlyBounds;
  const arma::mat& interclusterDistances;
  MetricType& metric;

  size_t distanceCalculations;

  TraversalInfoType traversalInfo;

  bool IsDescendantOf(const TreeType& potentialParent, const TreeType&
      potentialChild) const;
};

} // namespace kmeans
} // namespace mlpack

#include "dual_tree_kmeans_rules_impl.hpp"

#endif
