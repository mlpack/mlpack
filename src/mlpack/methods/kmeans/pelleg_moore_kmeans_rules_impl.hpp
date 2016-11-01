/**
 * @file pelleg_moore_kmeans_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the pruning rules and base cases necessary to perform
 * single-tree k-means clustering using the fast Pelleg-Moore k-means algorithm,
 * which has been shoehorned into the mlpack tree abstractions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_IMPL_HPP
#define MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "pelleg_moore_kmeans_rules.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
PellegMooreKMeansRules<MetricType, TreeType>::PellegMooreKMeansRules(
    const typename TreeType::Mat& dataset,
    const arma::mat& centroids,
    arma::mat& newCentroids,
    arma::Col<size_t>& counts,
    MetricType& metric) :
    dataset(dataset),
    centroids(centroids),
    newCentroids(newCentroids),
    counts(counts),
    metric(metric),
    distanceCalculations(0)
{
  // Nothing to do.
}

template<typename MetricType, typename TreeType>
inline force_inline
double PellegMooreKMeansRules<MetricType, TreeType>::BaseCase(
    const size_t /* queryIndex */,
    const size_t /* referenceIndex */)
{
  return 0.0;
}

template<typename MetricType, typename TreeType>
double PellegMooreKMeansRules<MetricType, TreeType>::Score(
    const size_t /* queryIndex */,
    TreeType& referenceNode)
{
  // Obtain the parent's blacklist.  If this is the root node, we'll start with
  // an empty blacklist.  This means that after each iteration, we don't need to
  // reset any statistics.
  if (referenceNode.Parent() == NULL ||
      referenceNode.Parent()->Stat().Blacklist().n_elem == 0)
    referenceNode.Stat().Blacklist().zeros(centroids.n_cols);
  else
    referenceNode.Stat().Blacklist() =
        referenceNode.Parent()->Stat().Blacklist();

  // The query index is a fake index that we won't use, and the reference node
  // holds all of the points in the dataset.  Our goal is to determine whether
  // or not this node is dominated by a single cluster.
  const size_t whitelisted = centroids.n_cols -
      arma::accu(referenceNode.Stat().Blacklist());

  distanceCalculations += whitelisted;

  // Which cluster has minimum distance to the node?
  size_t closestCluster = centroids.n_cols;
  double minMinDistance = DBL_MAX;
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    if (referenceNode.Stat().Blacklist()[i] == 0)
    {
      const double minDistance = referenceNode.MinDistance(centroids.col(i));
      if (minDistance < minMinDistance)
      {
        minMinDistance = minDistance;
        closestCluster = i;
      }
    }
  }

  // Now, for every other whitelisted cluster, determine if the closest cluster
  // owns the point.  This calculation is specific to hyperrectangle trees (but,
  // this implementation is specific to kd-trees, so that's okay).  For
  // circular-bound trees, the condition should be simpler and can probably be
  // expressed as a comparison between minimum and maximum distances.
  size_t newBlacklisted = 0;
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (referenceNode.Stat().Blacklist()[c] == 1 || c == closestCluster)
      continue;

    // This algorithm comes from the proof of Lemma 4 in the extended version
    // of the Pelleg-Moore paper (the CMU tech report, that is).  It has been
    // adapted for speed.
    arma::vec cornerPoint(centroids.n_rows);
    for (size_t d = 0; d < referenceNode.Bound().Dim(); ++d)
    {
      if (centroids(d, c) > centroids(d, closestCluster))
        cornerPoint(d) = referenceNode.Bound()[d].Hi();
      else
        cornerPoint(d) = referenceNode.Bound()[d].Lo();
    }

    const double closestDist = metric.Evaluate(cornerPoint,
        centroids.col(closestCluster));
    const double otherDist = metric.Evaluate(cornerPoint, centroids.col(c));

    distanceCalculations += 3; // One for cornerPoint, then two distances.

    if (closestDist < otherDist)
    {
      // The closest cluster dominates the node with respect to the cluster c.
      // So we can blacklist c.
      referenceNode.Stat().Blacklist()[c] = 1;
      ++newBlacklisted;
    }
  }

  if (whitelisted - newBlacklisted == 1)
  {
    // This node is dominated by the closest cluster.
    counts[closestCluster] += referenceNode.NumDescendants();
    newCentroids.col(closestCluster) += referenceNode.NumDescendants() *
        referenceNode.Stat().Centroid();

    return DBL_MAX;
  }

  // Perform the base case here.
  for (size_t i = 0; i < referenceNode.NumPoints(); ++i)
  {
    size_t bestCluster = centroids.n_cols;
    double bestDistance = DBL_MAX;
    for (size_t c = 0; c < centroids.n_cols; ++c)
    {
      if (referenceNode.Stat().Blacklist()[c] == 1)
        continue;

      ++distanceCalculations;

      // The reference index is the index of the data point.
      const double distance = metric.Evaluate(centroids.col(c),
          dataset.col(referenceNode.Point(i)));

      if (distance < bestDistance)
      {
        bestDistance = distance;
        bestCluster = c;
      }
    }

    // Add to resulting centroid.
    newCentroids.col(bestCluster) += dataset.col(referenceNode.Point(i));
    ++counts(bestCluster);
  }

  // Otherwise, we're not sure, so we can't prune.  Recursion order doesn't make
  // a difference, so we'll just return a score of 0.
  return 0.0;
}

template<typename MetricType, typename TreeType>
double PellegMooreKMeansRules<MetricType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const double oldScore)
{
  // There's no possible way that calling Rescore() can produce a prune now when
  // it couldn't before.
  return oldScore;
}

} // namespace kmeans
} // namespace mlpack

#endif
