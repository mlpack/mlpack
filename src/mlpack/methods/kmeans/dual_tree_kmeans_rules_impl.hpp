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
    const arma::mat& interclusterDistances,
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
    interclusterDistances(interclusterDistances),
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
  // Collect the number of clusters that have been pruned during the traversal.
  // The ternary operator may not be necessary.
  const size_t traversalPruned = (traversalInfo.LastReferenceNode() != NULL) ?
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
  // No pruning here, for now.
  return 0.0;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::Score(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // This won't happen with the root since it is explicitly set to 0.
  const size_t origPruned = referenceNode.Stat().ClustersPruned();
  if (referenceNode.Stat().ClustersPruned() == size_t(-1))
    referenceNode.Stat().ClustersPruned() =
        referenceNode.Parent()->Stat().ClustersPruned();

  traversalInfo.LastReferenceNode() = &referenceNode;

//  if (referenceNode.Begin() == 16954)
//    Log::Warn << "Visit r16954c" << referenceNode.Count() << ", q" <<
//queryNode.Begin() << "c" << queryNode.Count() << ".\n";

  // If there's no closest query node assigned, but the parent has one, take
  // that one.
  if (referenceNode.Stat().ClosestQueryNode() == NULL &&
      referenceNode.Parent() != NULL &&
      referenceNode.Parent()->Stat().ClosestQueryNode() != NULL)
  {
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "Update closest query node for r16954c" <<
//referenceNode.Count() << " to parent's, which is "
//          << ((TreeType*)
//referenceNode.Parent()->Stat().ClosestQueryNode())->Begin() << "c" <<
//((TreeType*) referenceNode.Parent()->Stat().ClosestQueryNode())->Count() <<
//".\n";

    referenceNode.Stat().ClosestQueryNode() =
        referenceNode.Parent()->Stat().ClosestQueryNode();
    referenceNode.Stat().MaxQueryNodeDistance() = std::min(
        referenceNode.Parent()->Stat().MaxQueryNodeDistance(),
        referenceNode.Stat().MaxQueryNodeDistance());
//    referenceNode.Stat().SecondClosestBound() = std::min(
//        referenceNode.Parent()->Stat().SecondClosestBound(),
//        referenceNode.Stat().SecondClosestBound());
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "Update second closest bound for r16954c" <<
//referenceNode.Count() << " to parent's, which "
//          << "is " << referenceNode.Stat().SecondClosestBound() << ".\n";
  }

  double score = HamerlyTypeScore(referenceNode);
  if (score == DBL_MAX)
  {
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "Hamerly prune for r16954c" << referenceNode.Count() << ", q" << queryNode.Begin() << "c" <<
//queryNode.Count() << ".\n";
    if (origPruned == size_t(-1))
    {
      const size_t cluster = referenceNode.Stat().Owner();
      newCentroids.col(cluster) += referenceNode.Stat().Centroid() *
          referenceNode.NumDescendants();
//      Log::Warn << "Hamerly prune: r" << referenceNode.Begin() << "c" <<
//          referenceNode.Count() << ".\n";
      counts(cluster) += referenceNode.NumDescendants();
      referenceNode.Stat().ClustersPruned() += queryNode.NumDescendants();
    }
    return DBL_MAX; // No other bookkeeping to do.
  }

  if (score != DBL_MAX)
  {
    score = ElkanTypeScore(queryNode, referenceNode);

    if (score != DBL_MAX)
    {
      // We also have to update things if the closest query node is null.  This
      // can probably be improved.
      const double minDistance = referenceNode.MinDistance(&queryNode);
      ++distanceCalculations;
      score = PellegMooreScore(queryNode, referenceNode, minDistance);
//      if (referenceNode.Begin() == 16954)
//        Log::Warn << "mQND for r16954c" << referenceNode.Count() << " is "
//            << referenceNode.Stat().MinQueryNodeDistance() << "; minDistance "
//            << minDistance << ", scb " <<
//referenceNode.Stat().SecondClosestBound() << ".\n";

      if (minDistance < referenceNode.Stat().MinQueryNodeDistance())
      {
        const double maxDistance = referenceNode.MaxDistance(&queryNode);
        if (!IsDescendantOf(*((TreeType*)
            referenceNode.Stat().ClosestQueryNode()), queryNode) &&
            referenceNode.Stat().MinQueryNodeDistance() != DBL_MAX &&
            referenceNode.Stat().MinQueryNodeDistance() <
            referenceNode.Stat().SecondClosestBound())
        {
          referenceNode.Stat().SecondClosestBound() =
              referenceNode.Stat().MinQueryNodeDistance();
          referenceNode.Stat().SecondClosestQueryNode() =
              referenceNode.Stat().ClosestQueryNode();
//          if (referenceNode.Begin() == 16954)
//            Log::Warn << "scb for r16954c" << referenceNode.Count() << " taken "
//                << "from minDistance, which is " <<
//referenceNode.Stat().MinQueryNodeDistance() << ".\n";
        }

        if (referenceNode.Stat().MinQueryNodeDistance() == DBL_MAX &&
            score == DBL_MAX &&
            minDistance < referenceNode.Stat().SecondClosestBound())
        {
          referenceNode.Stat().SecondClosestBound() = minDistance;
          referenceNode.Stat().SecondClosestQueryNode() = &queryNode;
//          if (referenceNode.Begin() == 16954)
//            Log::Warn << "scb for r16954c" << referenceNode.Count() << " taken "
//                << "from minDistance for pruned query node, which is " <<
//minDistance << ".\n";
        }

        if (score != DBL_MAX)
        {
          ++distanceCalculations;
          referenceNode.Stat().ClosestQueryNode() = (void*) &queryNode;
          referenceNode.Stat().MinQueryNodeDistance() = minDistance;
          referenceNode.Stat().MaxQueryNodeDistance() = maxDistance;

//          if (referenceNode.Begin() == 16954)
//            Log::Warn << "mQND for r16954c" << referenceNode.Count() << " updated to " << minDistance << " and "
//              << "MQND to " << maxDistance << " with furthest query node " <<
//              queryNode.Begin() << "c" << queryNode.Count() << ".\n";
        }
      }
      else if (IsDescendantOf(*((TreeType*)
          referenceNode.Stat().ClosestQueryNode()), queryNode))
      {
//        if (referenceNode.Begin() == 16954)
//          Log::Warn << "Old closest for r16954c" << referenceNode.Count() <<
//              " is q" << ((TreeType*)
//referenceNode.Stat().ClosestQueryNode())->Begin() << "c" << ((TreeType*)
//referenceNode.Stat().ClosestQueryNode())->Count() << " with mQND " <<
//referenceNode.Stat().MinQueryNodeDistance() << " and MQND " <<
//referenceNode.Stat().MaxQueryNodeDistance() << ".\n";
        const double maxDistance = referenceNode.MaxDistance(&queryNode);
        ++distanceCalculations;
        referenceNode.Stat().ClosestQueryNode() = (void*) &queryNode;
        referenceNode.Stat().MinQueryNodeDistance() = minDistance;
        referenceNode.Stat().MaxQueryNodeDistance() = maxDistance;

//        if (referenceNode.Begin() == 16954)
//          Log::Warn << "mQND for r16954c" << referenceNode.Count() << " updated to " << minDistance << " and "
//              << "MQND to " << maxDistance << " via descendant with fqn " <<
//              queryNode.Begin() << "c" << queryNode.Count() << ".\n";
      }
      else if (minDistance < referenceNode.Stat().SecondClosestBound())
      {
        referenceNode.Stat().SecondClosestBound() = minDistance;
        referenceNode.Stat().SecondClosestQueryNode() = &queryNode;
//        if (referenceNode.Begin() == 16954)
//          Log::Warn << "scb for r16954c" << referenceNode.Count() << " updated to " << minDistance << " via "
//              << queryNode.Begin() << "c" << queryNode.Count() << ".\n";
      }
    }
  }

//  if (((TreeType*) referenceNode.Stat().ClosestQueryNode())->NumDescendants() > 1)
//  {
//    referenceNode.Stat().SecondClosestBound() =
//        referenceNode.Stat().MinQueryNodeDistance();
//    referenceNode.Stat().SecondClosestQueryNode() =
//        referenceNode.Stat().ClosestQueryNode();
//  }

  if (score == DBL_MAX)
  {
    referenceNode.Stat().ClustersPruned() += queryNode.NumDescendants();
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "For r16954c" << referenceNode.Count() << ", q" <<
//queryNode.Begin() << "c" << queryNode.Count() << " is pruned.  Min distance is"
//    << " " << queryNode.MinDistance(&referenceNode) << " and scb is " <<
//referenceNode.Stat().SecondClosestBound() << ".\n";

    // Have we pruned everything?
    if (referenceNode.Stat().ClustersPruned() +
        visited[referenceNode.Descendant(0)] == centroids.n_cols)
    {
      for (size_t i = 0; i < referenceNode.NumDescendants(); ++i)
      {
        const size_t cluster = assignments[referenceNode.Descendant(i)];
        newCentroids.col(cluster) += dataset.col(referenceNode.Descendant(i));
        counts(cluster)++;
      }
    }
  }

  return score;
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
    TreeType& /* queryNode */,
    TreeType& /* referenceNode */,
    const double oldScore) const
{
  return oldScore;
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

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::HamerlyTypeScore(
    TreeType& referenceNode)
{
  if (referenceNode.Stat().HamerlyPruned())
  {
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "Hamerly prune! r" << referenceNode.Begin() << "c" <<
//referenceNode.Count() << ".\n";
    return DBL_MAX;
  }

  return 0.0;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::ElkanTypeScore(
    TreeType& queryNode,
    TreeType& referenceNode)
{
  // We have to calculate the minimum distance between the query node and the
  // reference node's best query node.  First, try to use the cached distance.
  if (queryNode.NumDescendants() > 1)
  {
    const double minQueryDistance = queryNode.Stat().FirstBound();
    const double score = ElkanTypeScore(queryNode, referenceNode,
        minQueryDistance);
    return score;
  }
  else
  {
    const double score = ElkanTypeScore(queryNode, referenceNode,
        interclusterDistances[queryNode.Descendant(0)]);
    return score;
  }
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::ElkanTypeScore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double minQueryDistance) const
{
  // See if we can do an Elkan-type prune on between-centroid distances.

  const double maxDistance = referenceNode.Stat().MaxQueryNodeDistance();
  if (maxDistance == DBL_MAX)
    return minQueryDistance;

  if ((minQueryDistance > 2.0 * maxDistance) &&
      !(IsDescendantOf(*(TreeType*) referenceNode.Stat().ClosestQueryNode(),
          queryNode)) &&
      (&queryNode != (TreeType*) referenceNode.Stat().ClosestQueryNode()))
  {
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "Elkan prune r16954c" << referenceNode.Count() << ", q" <<
//queryNode.Begin() << "c" << queryNode.Count() << "!\n";
    // Then we can conclude d_max(best(N_r), N_r) <= d_min(N_q, N_r) which
    // means that N_q cannot possibly hold any clusters that own any points in
    // N_r.
    return DBL_MAX;
  }

  return minQueryDistance;
}

template<typename MetricType, typename TreeType>
double DualTreeKMeansRules<MetricType, TreeType>::PellegMooreScore(
    TreeType& queryNode,
    TreeType& referenceNode,
    const double minDistance) const
{
  // If the minimum distance to the node is greater than the bound, then every
  // cluster in the query node cannot possibly be the nearest neighbor of any of
  // the points in the reference node.
//  if (referenceNode.Begin() == 16954)
//      Log::Warn << "Pelleg-Moore prune attempt r16954c" << referenceNode.Count() << ", "
//          << "q" << queryNode.Begin() << "c" << queryNode.Count() << "; "
//          << "minDistance " << minDistance << ", MQND " <<
//referenceNode.Stat().MaxQueryNodeDistance() << ".\n";
  if (minDistance > referenceNode.Stat().MaxQueryNodeDistance())
  {
//    if (referenceNode.Begin() == 16954)
//      Log::Warn << "Attempt successful!\n";
    return DBL_MAX;
  }

  return minDistance;
}

} // namespace kmeans
} // namespace mlpack

#endif
