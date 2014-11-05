/**
 * @file dual_tree_kmeans_rules_impl.hpp
 * @author Ryan Curtin
 *
 * A set of tree traversal rules for dual-tree k-means clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_RULES_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_kmeans_rules.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename TreeType>
DualTreeKMeansRules<MetricType, TreeType>::DualTreeKMeansRules(
    const typename TreeType::Mat& dataset,
    const arma::mat& centroids,
    arma::mat& newCentroids,
    arma::Col<size_t>& counts,
    const std::vector<size_t>& mappings,
    const size_t iteration,
    const arma::vec& clusterDistances,
    arma::vec& distances,
    arma::Col<size_t>& assignments,
    arma::Col<size_t>& distanceIteration,
    MetricType& metric) :
    dataset(dataset),
    centroids(centroids),
    newCentroids(newCentroids),
    counts(counts),
    mappings(mappings),
    iteration(iteration),
    clusterDistances(clusterDistances),
    distances(distances),
    assignments(assignments),
    distanceIteration(distanceIteration),
    metric(metric),
    distanceCalculations(0)
{
  // Nothing has been visited yet.
  visited.zeros(dataset.n_cols);
}

template<typename MetricType, typename TreeType>
inline force_inline double DualTreeKMeansRules<MetricType, TreeType>::BaseCase(
    const size_t queryIndex,
    const size_t referenceIndex)
{
//  Log::Info << "Base case, query " << queryIndex << " (" << mappings[queryIndex]
//      << "), reference " << referenceIndex << ".\n";

  // Collect the number of clusters that have been pruned during the traversal.
  // The ternary operator may not be necessary.
  const size_t traversalPruned = (traversalInfo.LastReferenceNode() != NULL &&
      traversalInfo.LastReferenceNode()->Stat().Iteration() == iteration) ?
      traversalInfo.LastReferenceNode()->Stat().ClustersPruned() : 0;

  // It's possible that the reference node has been pruned before we got to the
  // base case.  In that case, don't do the base case, and just return.
  if (traversalInfo.LastReferenceNode()->Stat().ClustersPruned() +
      visited[referenceIndex] == centroids.n_cols)
    return 0.0;

  ++distanceCalculations;

  const double distance = metric.Evaluate(centroids.col(queryIndex),
                                          dataset.col(referenceIndex));

  // Iteration change check.
  if (distanceIteration[referenceIndex] < iteration)
  {
    distanceIteration[referenceIndex] = iteration;
    distances[referenceIndex] = distance;
    assignments[referenceIndex] = mappings[queryIndex];
  }
  else if (distance < distances[referenceIndex])
  {
    distances[referenceIndex] = distance;
    assignments[referenceIndex] = mappings[queryIndex];
  }

  ++visited[referenceIndex];

  if (visited[referenceIndex] + traversalPruned == centroids.n_cols)
  {
//    Log::Warn << "Commit reference index " << referenceIndex << " to cluster "
//        << assignments[referenceIndex] << ".\n";
    newCentroids.col(assignments[referenceIndex]) +=
        dataset.col(referenceIndex);
    ++counts(assignments[referenceIndex]);
  }

  return distance;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::Score(
    const size_t queryIndex,
    TreeType& referenceNode)
{
  // Update from previous iteration, if necessary.
  IterationUpdate(referenceNode);

  // No pruning here, for now.
  return 0.0;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  IterationUpdate(referenceNode);

  traversalInfo.LastReferenceNode() = &referenceNode;

  // Can we update the minimum query node distance for this reference node?
  const double minDistance = referenceNode.MinDistance(&queryNode);
  ++distanceCalculations;
  if (minDistance < referenceNode.Stat().MinQueryNodeDistance())
  {
    referenceNode.Stat().ClosestQueryNode() = (void*) &queryNode;
    referenceNode.Stat().MinQueryNodeDistance() = minDistance;
    referenceNode.Stat().MaxQueryNodeDistance() =
        referenceNode.MaxDistance(&queryNode);
    ++distanceCalculations;
    return 0.0; // Pruning is not possible.
  }
  else if (IsDescendantOf(
      *((TreeType*) referenceNode.Stat().ClosestQueryNode()), queryNode))
  {
    // Just update.
    referenceNode.Stat().ClosestQueryNode() = (void*) &queryNode;
    referenceNode.Stat().MinQueryNodeDistance() = minDistance;
    referenceNode.Stat().MaxQueryNodeDistance() =
        referenceNode.MaxDistance(&queryNode);
    ++distanceCalculations;
    return 0.0; // Pruning is not possible.
  }

  // See if we can do an Elkan-type prune on between-centroid distances.
  const double maxDistance = referenceNode.Stat().MaxQueryNodeDistance();
  const double minQueryDistance = queryNode.MinDistance((TreeType*)
      referenceNode.Stat().ClosestQueryNode());
  ++distanceCalculations;

  if (minQueryDistance > 2.0 * maxDistance)
  {
    // Then we can conclude d_max(best(N_r), N_r) <= d_min(N_q, N_r) which
    // means that N_q cannot possibly hold any clusters that own any points in
    // N_r.
    referenceNode.Stat().ClustersPruned() += queryNode.NumDescendants();

    // Have we pruned everything?
    if (referenceNode.Stat().ClustersPruned() == centroids.n_cols - 1)
    {
      // Then the best query node must contain just one point.
      const TreeType* bestQueryNode = (TreeType*)
          referenceNode.Stat().ClosestQueryNode();
      const size_t cluster = mappings[bestQueryNode->Descendant(0)];

      referenceNode.Stat().Owner() = cluster;
      newCentroids.col(cluster) += referenceNode.NumDescendants() *
          referenceNode.Stat().Centroid();
      counts(cluster) += referenceNode.NumDescendants();
      referenceNode.Stat().ClustersPruned()++;
    }
    else if (referenceNode.Stat().ClustersPruned() +
        visited[referenceNode.Descendant(0)] == centroids.n_cols)
    {
      for (size_t i = 0; i < referenceNode.NumPoints(); ++i)
      {
        const size_t cluster = assignments[referenceNode.Point(i)];
        newCentroids.col(cluster) += dataset.col(referenceNode.Point(i));
        counts(cluster)++;
      }
    }

    return DBL_MAX;
  }

  return minQueryDistance;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::Rescore(
    const size_t /* queryIndex */,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  return oldScore;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::Rescore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double oldScore) const
{
  if (oldScore == DBL_MAX)
    return oldScore; // We can't unprune something.  This shouldn't happen.

  // Can we update the minimum query node distance for this reference node?
  const double minQueryDistance = oldScore;

  // See if we can do an Elkan-type prune on between-centroid distances.
  const double maxDistance = referenceNode.Stat().MaxQueryNodeDistance();

  if (minQueryDistance > 2.0 * maxDistance)
  {
    // Then we can conclude d_max(best(N_r), N_r) <= d_min(N_q, N_r) which
    // means that N_q cannot possibly hold any clusters that own any points in
    // N_r.
    referenceNode.Stat().ClustersPruned() += queryNode.NumDescendants();

    // Have we pruned everything?
    if (referenceNode.Stat().ClustersPruned() == centroids.n_cols - 1)
    {
      // Then the best query node must contain just one point.
      const TreeType* bestQueryNode = (TreeType*)
          referenceNode.Stat().ClosestQueryNode();
      const size_t cluster = mappings[bestQueryNode->Descendant(0)];

      referenceNode.Stat().Owner() = cluster;
      newCentroids.col(cluster) += referenceNode.NumDescendants() *
          referenceNode.Stat().Centroid();
      counts(cluster) += referenceNode.NumDescendants();
      referenceNode.Stat().ClustersPruned()++;
    }
    else if (referenceNode.Stat().ClustersPruned() +
        visited[referenceNode.Descendant(0)] == centroids.n_cols)
    {
      for (size_t i = 0; i < referenceNode.NumPoints(); ++i)
      {
        const size_t cluster = assignments[referenceNode.Point(i)];
        newCentroids.col(cluster) += dataset.col(referenceNode.Point(i));
        counts(cluster)++;
      }
    }

    return DBL_MAX;
  }

  return oldScore;
}

template<typename MetricType, typename TreeType>
inline size_t DualTreeKMeansRules<MetricType, TreeType>::IterationUpdate(
    TreeType& referenceNode) const
{
  if (referenceNode.Stat().Iteration() == iteration)
    return 0;

  referenceNode.Stat().Iteration() = iteration;
  referenceNode.Stat().ClustersPruned() = (referenceNode.Parent() == NULL) ?
      0 : referenceNode.Parent()->Stat().ClustersPruned();
  referenceNode.Stat().ClosestQueryNode() = (referenceNode.Parent() == NULL) ?
      NULL : referenceNode.Parent()->Stat().ClosestQueryNode();

  if (referenceNode.Stat().ClosestQueryNode() != NULL)
    referenceNode.Stat().MinQueryNodeDistance() =
        referenceNode.MinDistance((TreeType*)
        referenceNode.Stat().ClosestQueryNode());

  const size_t itDiff = iteration - referenceNode.Stat().Iteration();
  if (itDiff > 1)
  {
    // Maybe this can be tighter?
    referenceNode.Stat().MinQueryNodeDistance() = DBL_MAX;
  }
  else
  {
    if (referenceNode.Stat().MinQueryNodeDistance() != DBL_MAX)
    {
      // Update the distance to the closest query node.  If this node has an
      // owner, we know how far to increase the bound.  Otherwise, increase it
      // by the furthest amount that any centroid moved.
//      if (referenceNode.Stat().Owner() < centroids.n_cols)
//        referenceNode.Stat().MinQueryNodeDistance() +=
//            clusterDistances(referenceNode.Stat().Owner());
//      else
//        referenceNode.Stat().MinQueryNodeDistance() = DBL_MAX;
//            clusterDistances(centroids.n_cols);
      if (referenceNode.Stat().MaxQueryNodeDistance() == DBL_MAX)
        referenceNode.Stat().MinQueryNodeDistance() = DBL_MAX;
      else
      {
        referenceNode.Stat().MinQueryNodeDistance() +=
            clusterDistances(centroids.n_cols);
//referenceNode.Stat().MaxQueryNodeDistance() +
//clusterDistances(centroids.n_cols);
      }
    }
  }

  return 1;
}

template<typename MetricType, typename TreeType>
bool DualTreeKMeansRules<MetricType, TreeType>::IsDescendantOf(
    const TreeType& potentialParent,
    const TreeType& potentialChild) const
{
  if (potentialChild.Parent() == &potentialParent)
    return true;
  else if (potentialChild.Parent() == NULL)
    return false;
  else
    return IsDescendantOf(potentialParent, *potentialChild.Parent());
}

} // namespace kmeans
} // namespace mlpack

#endif
