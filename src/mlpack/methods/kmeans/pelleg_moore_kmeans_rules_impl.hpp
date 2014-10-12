/**
 * @file pelleg_moore_kmeans_rules_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the pruning rules and base cases necessary to perform
 * single-tree k-means clustering using the fast Pelleg-Moore k-means algorithm,
 * which has been shoehorned into the mlpack tree abstractions.
 */
#ifndef __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_IMPL_HPP

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
    baseCases(0),
    scores(0),
    spareBlacklist(centroids.n_cols)
{
  // Nothing to do.
  spareBlacklist.zeros();
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
  arma::uvec* blacklistPtr = NULL;
  if (referenceNode.Parent() == NULL ||
      referenceNode.Parent()->Stat().Blacklist().size() == 0)
    blacklistPtr = &spareBlacklist;
  else
    blacklistPtr = &referenceNode.Parent()->Stat().Blacklist();

  // If the blacklist hasn't been initialized, fill it with zeros.
  if (blacklistPtr->n_elem == 0)
    blacklistPtr->zeros(centroids.n_cols);
  referenceNode.Stat().Blacklist() = *blacklistPtr;

  // The query index is a fake index that we won't use, and the reference node
  // holds all of the points in the dataset.  Our goal is to determine whether
  // or not this node is dominated by a single cluster.
  const size_t whitelisted = centroids.n_cols - arma::accu(*blacklistPtr);

  scores += whitelisted;

  arma::vec minDistances(whitelisted);
  minDistances.fill(DBL_MAX);
  arma::Col<size_t> indexMappings(whitelisted);
  size_t index = 0;
  for (size_t i = 0; i < centroids.n_cols; ++i)
  {
    if ((*blacklistPtr)[i] == 0)
    {
      minDistances(index) = referenceNode.MinDistance(centroids.col(i));
      indexMappings(index) = i;
      ++index;
    }
  }

  // Which cluster has minimum distance to the node?  Sort by distance.
  // This should probably be rewritten -- we only need the minimum, not the
  // entire sorted list.  That'll cost O(k) not O(k log k) (depending on sort
  // type).
  arma::uvec sortedClusterIndices = sort_index(minDistances);
  const size_t closestCluster = indexMappings(sortedClusterIndices[0]);


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
    // This node is dominated by the first cluster.
    const size_t cluster = indexMappings(sortedClusterIndices[0]);
    counts[cluster] += referenceNode.NumDescendants();
    newCentroids.col(cluster) += referenceNode.NumDescendants() *
        referenceNode.Stat().Centroid();

    if (referenceNode.Parent() == NULL ||
        referenceNode.Parent()->Stat().Blacklist().size() == 0)
    {
      spareBlacklist.zeros(centroids.n_cols);
    }

    return DBL_MAX;
  }

  if (referenceNode.Parent() == NULL ||
      referenceNode.Parent()->Stat().Blacklist().size() == 0)
  {
    spareBlacklist.zeros(centroids.n_cols);
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

      ++baseCases;

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

}; // namespace kmeans
}; // namespace mlpack

#endif
