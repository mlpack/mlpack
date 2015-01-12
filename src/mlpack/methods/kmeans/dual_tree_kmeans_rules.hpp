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
                      arma::Col<size_t>& distanceIteration,
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
  arma::Col<size_t> visited;
  arma::Col<size_t>& distanceIteration;
  const arma::mat& interclusterDistances;
  MetricType& metric;

  size_t distanceCalculations;

  TraversalInfoType traversalInfo;

  bool IsDescendantOf(const TreeType& potentialParent, const TreeType&
      potentialChild) const;

  /**
   * See if an Elkan-type prune can be performed.  If so, return DBL_MAX;
   * otherwise, return a score.  The Elkan-type prune can occur when the minimum
   * distance between the query node and the current best query node for the
   * reference node (referenceNode.Stat().ClosestQueryNode()) is greater than
   * two times the maximum distance between the reference node and the current
   * best query node (again, referenceNode.Stat().ClosestQueryNode()).
   *
   * @param queryNode Query node.
   * @param referenceNode Reference node.
   */
  double ElkanTypeScore(TreeType& queryNode, TreeType& referenceNode);

  /**
   * See if an Elkan-type prune can be performed.  If so, return DBL_MAX;
   * otherwise, return a score.  The Elkan-type prune can occur when the minimum
   * distance between the query node and the current best query node for the
   * reference node (referenceNode.Stat().ClosestQueryNode()) is greater than
   * two times the maximum distance between the reference node and the current
   * best query node (again, referenceNode.Stat().ClosestQueryNode()).
   *
   * This particular overload is for when the minimum distance between the query
   * noed and the current best query node has already been calculated.
   *
   * @param queryNode Query node.
   * @param referenceNode Reference node.
   * @param minQueryDistance Minimum distance between query node and current
   *      best query node for the reference node.
   */
  double ElkanTypeScore(TreeType& queryNode,
                        TreeType& referenceNode,
                        const double minQueryDistance) const;

  double PellegMooreScore(TreeType& /* queryNode */,
                          TreeType& referenceNode,
                          const double minDistance) const;
};

} // namespace kmeans
} // namespace mlpack

#include "dual_tree_kmeans_rules_impl.hpp"

#endif
