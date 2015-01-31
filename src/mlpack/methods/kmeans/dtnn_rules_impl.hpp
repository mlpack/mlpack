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
    arma::Mat<size_t>& neighbors,
    arma::mat& distances,
    MetricType& metric) :
    rules(centroids, dataset, neighbors, distances, metric)
{
  // Nothing to do.
}

template<typename MetricType, typename TreeType>
inline force_inline double DTNNKMeansRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
  // We'll check if the query point has been Hamerly pruned.  If so, don't
  // continue.

  return rules.BaseCase(queryIndex, referenceIndex);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  return rules.Score(queryIndex, referenceNode);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  if (queryNode.Stat().Pruned())
    return DBL_MAX;

  // Check if the query node is Hamerly pruned, and if not, then don't continue.
  return rules.Score(queryNode, referenceNode);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Rescore(
    const size_t queryIndex,
    TreeType& referenceNode,
    const double oldScore)
{
  return rules.Rescore(queryIndex, referenceNode, oldScore);
}

template<typename MetricType, typename TreeType>
inline double DTNNKMeansRules<MetricType, TreeType>::Rescore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double oldScore)
{
  // No need to check for a Hamerly prune.  Because we've already done that in
  // Score().
  return rules.Rescore(queryNode, referenceNode, oldScore);
}

} // namespace kmeans
} // namespace mlpack

#endif
