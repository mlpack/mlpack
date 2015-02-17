/**
 * @file dtnn_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of DualTreeKMeansRules.
 */
#ifndef __MLPACK_METHODS_KMEANS_DTNN_RULES_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_DTNN_RULES_IMPL_HPP

#include "dtnn_rules.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
DTNNKMeansRules<MetricType, TreeType>::DTNNKMeansRules(
    const arma::mat& centroids,
    const arma::mat& dataset,
    arma::Col<size_t>& assignments,
    arma::vec& upperBounds,
    arma::vec& lowerBounds,
    MetricType& metric,
    const std::vector<bool>& prunedPoints,
    const std::vector<size_t>& oldFromNewCentroids,
    std::vector<bool>& visited) :
    centroids(centroids),
    dataset(dataset),
    assignments(assignments),
    upperBounds(upperBounds),
    lowerBounds(lowerBounds),
    metric(metric),
    prunedPoints(prunedPoints),
    oldFromNewCentroids(oldFromNewCentroids),
    visited(visited),
    baseCases(0),
    scores(0)
{
  // Nothing to do.
}

template<typename MetricType, typename TreeType>
inline force_inline double DTNNKMeansRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  if (prunedPoints[queryIndex])
    return 0.0; // Returning 0 shouldn't be a problem.

  // Any base cases imply that we will get a result.
  visited[queryIndex] = true;

  // Calculate the distance.
  ++baseCases;
  const double distance = metric.Evaluate(dataset.col(queryIndex),
                                          centroids.col(referenceIndex));

  if (distance < upperBounds[queryIndex])
  {
    lowerBounds[queryIndex] = upperBounds[queryIndex];
    upperBounds[queryIndex] = distance;
    assignments[queryIndex] = referenceIndex;
  }
  else if (distance < lowerBounds[queryIndex])
  {
    lowerBounds[queryIndex] = distance;
  }

  return distance;
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& /* referenceNode */)
{
  // If the query point has already been pruned, then don't recurse further.
  if (prunedPoints[queryIndex])
    return DBL_MAX;

  // No pruning at this level (for now).
  return 0;
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // Pruned() for the root node must never be set to size_t(-1).
  if (queryNode.Stat().Pruned() == size_t(-1))
  {
    queryNode.Stat().Pruned() = queryNode.Parent()->Stat().Pruned();
    queryNode.Stat().LowerBound() = queryNode.Parent()->Stat().LowerBound();
    queryNode.Stat().Owner() = queryNode.Parent()->Stat().Owner();
  }

  if (queryNode.Stat().Pruned() == centroids.n_cols)
  {
    return DBL_MAX;
  }

  // Get minimum and maximum distances.
  math::Range distances = queryNode.RangeDistance(&referenceNode);
  double score = distances.Lo();
  ++scores;
  if (distances.Lo() > queryNode.Stat().UpperBound() ||
      distances.Lo() > queryNode.Stat().LastUpperBound())
  {
    // The reference node can own no points in this query node.  We may improve
    // the lower bound on pruned nodes, though.
    if (distances.Lo() < queryNode.Stat().LowerBound())
      queryNode.Stat().LowerBound() = distances.Lo();

    // This assumes that reference clusters don't appear elsewhere in the tree.
    queryNode.Stat().Pruned() += referenceNode.NumDescendants();
    score = DBL_MAX;
  }
  else if (distances.Hi() < queryNode.Stat().UpperBound())
  {
    // We can improve the best estimate.
    queryNode.Stat().UpperBound() = distances.Hi();
    // If this node has only one descendant, then it may be the owner.
    if (referenceNode.NumDescendants() == 1)
      queryNode.Stat().Owner() = referenceNode.Descendant(0);
  }

  // Is everything pruned?
  if (queryNode.Stat().Pruned() == centroids.n_cols - 1)
  {
    queryNode.Stat().Pruned() = centroids.n_cols; // Owner() is already set.
    return DBL_MAX;
  }

  return score;
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const double oldScore)
{
  // No rescoring (for now).
  return oldScore;
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Rescore(
    TreeType& /* queryNode */,
    TreeType& /* referenceNode */,
    const double oldScore)
{
  // No rescoring (for now).
  return oldScore;
}

} // namespace kmeans
} // namespace mlpack

#endif
