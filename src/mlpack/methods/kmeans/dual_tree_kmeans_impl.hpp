/**
 * @file dual_tree_kmeans_impl.hpp
 * @author Ryan Curtin
 *
 * A dual-tree algorithm for a single k-means iteration.
 */
#ifndef __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_kmeans.hpp"
#include "dual_tree_kmeans_rules.hpp"

#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename MatType, typename TreeType>
DualTreeKMeans<MetricType, MatType, TreeType>::DualTreeKMeans(
    const MatType& dataset,
    MetricType& metric) :
    datasetOrig(dataset),
    dataset(tree::TreeTraits<TreeType>::RearrangesDataset ? datasetCopy :
        datasetOrig),
    metric(metric),
    iteration(0),
    distanceCalculations(0)
{
  distances.set_size(dataset.n_cols);
  distances.fill(DBL_MAX);
  assignments.zeros(dataset.n_cols);
  distanceIteration.zeros(dataset.n_cols);

  Timer::Start("tree_building");

  // Copy the dataset, if necessary.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
    datasetCopy = datasetOrig;

  // Now build the tree.  We don't need any mappings.
  tree = new TreeType(const_cast<typename TreeType::Mat&>(this->dataset));

  Timer::Stop("tree_building");
}

template<typename MetricType, typename MatType, typename TreeType>
DualTreeKMeans<MetricType, MatType, TreeType>::~DualTreeKMeans()
{
  if (tree)
    delete tree;
}

template<typename MetricType, typename MatType, typename TreeType>
double DualTreeKMeans<MetricType, MatType, TreeType>::Iterate(
    const arma::mat& centroids,
    arma::mat& newCentroids,
    arma::Col<size_t>& counts)
{
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);
  if (clusterDistances.n_elem != centroids.n_cols + 1)
  {
    clusterDistances.set_size(centroids.n_cols + 1);
    clusterDistances.fill(DBL_MAX / 2.0); // To prevent overflow.
  }

  // Build a tree on the centroids.
  arma::mat oldCentroids(centroids);
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree = BuildTree<TreeType>(
      const_cast<typename TreeType::Mat&>(centroids), oldFromNewCentroids);
  for (size_t i = 0; i < oldFromNewCentroids.size(); ++i)
    Log::Warn << oldFromNewCentroids[i] << " ";
  Log::Warn << "\n";

  // Now calculate distances between centroids.
  neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType, TreeType>
      nns(centroidTree, centroids);
  arma::mat interclusterDistances;
  arma::Mat<size_t> closestClusters; // We don't actually care about these.
  nns.Search(1, closestClusters, interclusterDistances);

  distanceCalculations += nns.BaseCases();
  distanceCalculations += nns.Scores();

  // Update FirstBound().
  ClusterTreeUpdate(centroidTree, interclusterDistances);

  // Now run the dual-tree algorithm.
  typedef DualTreeKMeansRules<MetricType, TreeType> RulesType;
  RulesType rules(dataset, centroids, newCentroids, counts, oldFromNewCentroids,
      iteration, clusterDistances, distances, assignments, distanceIteration,
      interclusterDistances, metric);

  // Use the dual-tree traverser.
//typename TreeType::template DualTreeTraverser<RulesType> traverser(rules);
  typename TreeType::template BreadthFirstDualTreeTraverser<RulesType>
      traverser(rules);

  tree->Stat().ClustersPruned() = 0; // The constructor sets this to -1.
  traverser.Traverse(*centroidTree, *tree);

  distanceCalculations += rules.DistanceCalculations();

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  clusterDistances.zeros();
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts[c] == 0)
    {
      newCentroids.col(c).fill(DBL_MAX); // Should have happened anyway I think.
    }
    else
    {
      const size_t oldCluster = oldFromNewCentroids[c];
      newCentroids.col(oldCluster) /= counts(oldCluster);
      const double dist = metric.Evaluate(centroids.col(c),
                                          newCentroids.col(oldCluster));
      if (dist > clusterDistances[centroids.n_cols])
        clusterDistances[centroids.n_cols] = dist;
      clusterDistances[oldCluster] = dist;
      residual += std::pow(dist, 2.0);
    }
  }

  // Update the tree with the centroid movement information.
  size_t hamerlyPruned = 0;
  TreeUpdate(tree, centroids.n_cols, clusterDistances, assignments,
      oldCentroids, dataset, oldFromNewCentroids, hamerlyPruned);

  delete centroidTree;

  ++iteration;
  return std::sqrt(residual);
}

template<typename MetricType, typename MatType, typename TreeType>
void DualTreeKMeans<MetricType, MatType, TreeType>::ClusterTreeUpdate(
    TreeType* node,
    const arma::mat& distances)
{
  // Just update the first bound, after recursing to the bottom.
  double firstBound = 0.0;
  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    ClusterTreeUpdate(&node->Child(i), distances);
    if (node->Child(i).Stat().FirstBound() >= firstBound)
      firstBound = node->Child(i).Stat().FirstBound();
  }
  for (size_t i = 0; i < node->NumPoints(); ++i)
  {
    if (distances(0, node->Point(i)) > firstBound)
      firstBound = distances(0, node->Point(i));
  }

  node->Stat().FirstBound() = firstBound;
}

template<typename TreeType>
bool IsDescendantOf(
    const TreeType& potentialParent,
    const TreeType& potentialChild)
{
  if (potentialChild.Parent() == &potentialParent)
    return true;
  else if (potentialChild.Parent() == NULL)
    return false;
  else
    return IsDescendantOf(potentialParent, *potentialChild.Parent());
}

template<typename MetricType, typename MatType, typename TreeType>
void DualTreeKMeans<MetricType, MatType, TreeType>::TreeUpdate(
    TreeType* node,
    const size_t clusters,
    const arma::vec& clusterDistances,
    const arma::Col<size_t>& assignments,
    const arma::mat& centroids,
    const arma::mat& dataset,
    const std::vector<size_t>& oldFromNew,
    size_t& hamerlyPruned)
{
  // This is basically IterationUpdate(), but pulled out to be separate from the
  // actual dual-tree algorithm.

  if (node->Parent() != NULL && node->Parent()->Stat().Owner() < clusters)
    node->Stat().Owner() = node->Parent()->Stat().Owner();

  const size_t cluster = assignments[node->Descendant(0)];
//  if (node->Begin() == 16954)
//    Log::Info << "r16954c" << node->Count() << ", descendant 0 has cluster "
//        << cluster << ".\n";
  bool allSame = true;
  for (size_t i = 1; i < node->NumDescendants(); ++i)
  {
//    if (node->Begin() == 16954)
//      Log::Info << "Descendant " << i << " has cluster " <<
//          assignments[node->Descendant(i)] << ".\n";
    if (assignments[node->Descendant(i)] != cluster)
    {
      allSame = false;
      break;
    }
  }

  if (allSame)
    node->Stat().Owner() = cluster;
  else
    node->Stat().Owner() = clusters; // No owner.

  const bool prunedLastIteration = node->Stat().HamerlyPruned();
  node->Stat().HamerlyPruned() = false;

  // The easy case: this node had an owner.
  if (node->Stat().Owner() < clusters)
  {
    // During the last iteration, this node was pruned.
    const size_t owner = node->Stat().Owner();
    if (node->Stat().MaxQueryNodeDistance() != DBL_MAX)
      node->Stat().MaxQueryNodeDistance() += clusterDistances[owner];
    if (node->Stat().MinQueryNodeDistance() != DBL_MAX)
      node->Stat().MinQueryNodeDistance() += clusterDistances[owner];

    // Check if we can perform a Hamerly prune: if the node has an owner, and
    // the second closest cluster could not have moved close enough that any
    // points could have changed assignment, then this node *must* belong to the
    // same owner in the next iteration.  Note that MaxQueryNodeDistance() has
    // already been adjusted for cluster movement.

    // Re-set second closest bound if necessary.
    if (node->Stat().SecondClosestBound() == DBL_MAX && node->Parent() == NULL)
      node->Stat().SecondClosestBound() = 0.0; // Don't prune the root.

//    if (node->Begin() == 16954)
//      Log::Warn << "r16954c" << node->Count() << " scb " <<
//node->Stat().SecondClosestBound() << " and lscb " <<
//node->Stat().LastSecondClosestBound() << ".\n";

    // If both the second closest bound and last second closest bound are valid,
    // we have the option of taking the better of the two bounds.  But if only
    // one is valid, take the minimum of the two (which will be the valid one).
    // If neither is valid, then we end up with a second closest bound of
    // DBL_MAX.
    const double scb = node->Stat().SecondClosestBound();
    const double lscb = node->Stat().LastSecondClosestBound();
    if (scb != DBL_MAX && lscb != DBL_MAX)
      node->Stat().SecondClosestBound() = std::max(scb, lscb);
    else
      node->Stat().SecondClosestBound() = std::min(scb, lscb);

    // But if we were Hamerly pruned last time, we can't trust the second
    // closest bound and thus have to take last iteration's.
    if (prunedLastIteration)
      node->Stat().SecondClosestBound() = lscb;
    else
    {
      // Now, we must ensure that we don't need to take the parent's second
      // closest bound.  We surely do if the current bound is DBL_MAX.  We
      // already took care of the root node earlier so we don't need to check if
      // Parent() is NULL.
      if (node->Stat().SecondClosestBound() == DBL_MAX)
        node->Stat().SecondClosestBound() =
            node->Parent()->Stat().SecondClosestBound();

      // There may exist a case where the true second closest query node got
      // pruned by the parent, and was thus never visited with this node.  This
      // situation occurs if the second closest query node is not a descendant
      // of the second closest query node of the parent.
//      if (node->Stat().SecondClosestQueryNode() != NULL)
//      {
//        if (node->Begin() == 16954)
//        {
//          Log::Warn << "Second closest query node is q" << ((TreeType*)
//node->Stat().SecondClosestQueryNode())->Begin() << "c" << ((TreeType*)
//node->Stat().SecondClosestQueryNode())->Count() << ", with scb " <<
//node->Stat().SecondClosestBound() << ".\n";
//          Log::Warn << "True SCB to this node should be " <<
//node->MinDistance((TreeType*) node->Stat().SecondClosestQueryNode()) << ".\n";
//        }
//      }

//      if (node->Stat().ClosestQueryNode() != NULL)
//        if (node->Begin() == 16954)
//          Log::Warn << "Closest query node: q" << ((TreeType*)
//node->Stat().ClosestQueryNode())->Begin() << "c" << ((TreeType*)
//node->Stat().ClosestQueryNode())->Count() << ", with MQND " <<
//node->Stat().MaxQueryNodeDistance() << " and mQND " <<
//node->Stat().MinQueryNodeDistance() << ".\n";

      // If the closest query node contains more than one descendant, we have to
      // find the closest...
      TreeType* cqn = (TreeType*) node->Stat().ClosestQueryNode();
      if (cqn != NULL && cqn->NumDescendants() > 1)
      {
        size_t closest = centroids.n_cols;
        double closestDistance = DBL_MAX;
        size_t secondClosest = centroids.n_cols;
        double secondClosestDistance = DBL_MAX;
        for (size_t i = 0; i < cqn->NumDescendants(); ++i)
        {
          const size_t index = cqn->Descendant(i);
          const double distance =
              node->MinDistance(centroids.col(oldFromNew[index]));
//        Log::Info << "Index " << index << ", distance " << distance << " (i "
//            << i + cqn->Begin() << ").\n";
          ++distanceCalculations;
          if (distance < closestDistance)
          {
            secondClosest = closest;
            secondClosestDistance = closestDistance;
            closest = index;
            closestDistance = distance;
          }
          else if (distance < secondClosestDistance)
          {
            secondClosest = index;
            secondClosestDistance = distance;
          }
        }
  
        // Recalculate maximum distance.
        const double maxDistance = node->MaxDistance(centroids.col(closest));
        ++distanceCalculations;

        node->Stat().MinQueryNodeDistance() = closestDistance;
        node->Stat().MaxQueryNodeDistance() = maxDistance;
        if (secondClosestDistance < node->Stat().SecondClosestBound())
          node->Stat().SecondClosestBound() = secondClosestDistance;

//      if (node->Begin() == 16954)
//        Log::Warn << "After recalculation, closest for r" << node->Begin() << "c" << node->Count()
//<< " is " << closest << ", with mQND " << node->Stat().MinQueryNodeDistance() <<
//", MQND" << node->Stat().MaxQueryNodeDistance() << ", and scb " <<
//node->Stat().SecondClosestBound() << ", " << secondClosest << ".\n";
      }

      if (node->Parent() != NULL &&
node->Parent()->Stat().SecondClosestQueryNode() != NULL)
//        if (node->Begin() == 16954)
//          Log::Warn << "Parent's (r" << node->Parent()->Begin() << "c"
//<< node->Parent()->Count() << ") second closest query node is q" << ((TreeType*)
//node->Parent()->Stat().SecondClosestQueryNode())->Begin() << "c" << ((TreeType*)
//node->Parent()->Stat().SecondClosestQueryNode())->Count() << ", with scb " <<
//node->Parent()->Stat().SecondClosestBound() << ".\n";

      if (node->Parent() != NULL && 
          node->Stat().SecondClosestQueryNode() != NULL &&
          node->Parent()->Stat().SecondClosestQueryNode() != NULL && !IsDescendantOf((*(TreeType*)
node->Parent()->Stat().SecondClosestQueryNode()), (*(TreeType*)
node->Stat().SecondClosestQueryNode())))
      {
        // Does this even work?
        const double minDistance = node->MinDistance((TreeType*)
            node->Parent()->Stat().SecondClosestQueryNode());
//        if (node->Begin() == 16954)
//          Log::Warn << "r16954c" << node->Count() << " has min distance " <<
//minDistance << " to SCQN of parent (q" << ((TreeType*)
//node->Parent()->Stat().SecondClosestQueryNode())->Begin() << "c" << ((TreeType*)
//node->Parent()->Stat().SecondClosestQueryNode())->Count() << ")\n";

        if (minDistance < node->Stat().SecondClosestBound())
        {
          node->Stat().SecondClosestBound() =
              node->Parent()->Stat().SecondClosestBound();
          node->Stat().SecondClosestQueryNode() =
              node->Parent()->Stat().SecondClosestQueryNode();
//          if (node->Begin() == 16954)
//            Log::Warn << "r16954c" << node->Count() << " has recalculated scb "
//                << node->Stat().SecondClosestBound() << ".\n";
        }
      }
    }

//    if (node->Begin() == 16954)
//      Log::Warn << "r16954c" << node->Count() << " now scb " <<
//node->Stat().SecondClosestBound() << " and lscb " <<
//node->Stat().LastSecondClosestBound() << ".\n";
//      if (node->Begin() == 16954)
//        Log::Warn << "Update second closest bound for r16954c" <<
//node->Count() << " to " << node->Stat().SecondClosestBound() << ", which could "
//      << "have been parent's (" << node->Parent()->Stat().SecondClosestBound()
//<< ") or adjusted last iteration's (" << node->Stat().LastSecondClosestBound()
//<< ").\n";

//    if (node->Begin() == 16954)
//      Log::Warn << "r16954c" << node->Count() << " has second bound " <<
//node->Stat().SecondClosestBound() << " (q" << ((TreeType*)
//node->Stat().SecondClosestQueryNode())->Begin() << "c" << ((TreeType*)
//node->Stat().SecondClosestQueryNode())->Count() << ") and parent has second "
//          << "bound " << node->Parent()->Stat().SecondClosestBound() << " (q"
//          << ((TreeType*)
//node->Parent()->Stat().SecondClosestQueryNode())->Begin() << "c" << ((TreeType*)
//node->Parent()->Stat().SecondClosestQueryNode())->Count() << ").\n";
/*
    if (node->Parent() != NULL &&
node->Parent()->Stat().SecondClosestQueryNode() != NULL &&
node->Stat().SecondClosestQueryNode() != NULL && !IsDescendantOf(*((TreeType*)
node->Stat().SecondClosestQueryNode()), *((TreeType*)
node->Parent()->Stat().SecondClosestQueryNode())) &&
node->Parent()->Stat().SecondClosestBound() < node->Stat().SecondClosestBound())
    {
//      if (node->Begin() == 16954)
//        Log::Warn << "Take second closest bound for r16954c" <<
//node->Count() << " from parent: " << node->Parent()->Stat().SecondClosestBound()
//<< " (was " << node->Stat().SecondClosestBound() << ").\n";
          node->Stat().SecondClosestBound() =
node->Parent()->Stat().SecondClosestBound();
    }*/

//    if (node->Begin() == 16954)
//    {
//      Log::Warn << "Attempt Hamerly prune on r16954c" << node->Count() <<
//          " with MQND " << node->Stat().MaxQueryNodeDistance() << ", scb "
//          << node->Stat().SecondClosestBound() << ", owner " <<
//node->Stat().Owner() << ", and clusterDistances " << clusterDistances[clusters]
//<< ".\n";
//    }

    // Check the second bound.  (This is time-consuming...)
    for (size_t j = 0; j < node->NumDescendants(); ++j)
    {
      arma::vec distances(centroids.n_cols);
      double secondClosestDist = DBL_MAX;
      for (size_t i = 0; i < centroids.n_cols; ++i)
      {
        const double distance = MetricType::Evaluate(centroids.col(i),
            dataset.col(node->Descendant(j)));
        if (distance < secondClosestDist && i != node->Stat().Owner())
          secondClosestDist = distance;

        distances(i) = distance;
      }

//        if (node->Begin() == 16954)
//          Log::Warn << "r16954c" << node->Count() << ": " << distances.t();
      if (secondClosestDist < node->Stat().SecondClosestBound() - 1e-15)
      {
        Log::Warn << "r" << node->Begin() << "c" << node->Count() << ":\n";
        Log::Warn << "Owner " << node->Stat().Owner() << ", mqnd " <<
node->Stat().MaxQueryNodeDistance() << ", mnqnd " <<
node->Stat().MinQueryNodeDistance() << ".\n";
        Log::Warn << distances.t();
        Log::Fatal << "Second closest bound " <<
node->Stat().SecondClosestBound() << " is too loose! -- " << secondClosestDist
            << "! (" << node->Stat().SecondClosestBound() - secondClosestDist
<< ")\n";
      }
    }

    if (node->Stat().MaxQueryNodeDistance() < node->Stat().SecondClosestBound()
        - clusterDistances[clusters])
    {
      node->Stat().HamerlyPruned() = true;
//      if (node->Begin() == 16954)
      if (!node->Parent()->Stat().HamerlyPruned())
      {
//        Log::Warn << "Mark r" << node->Begin() << "c" << node->Count() << " as "
//            << "Hamerly pruned.\n";
        hamerlyPruned += node->NumDescendants();
      }
    }
//    else
//    {
//      Log::Warn << "Failed Hamerly prune for r" << node->Begin() << "c" <<
//          node->Count() << "; mqnd " << node->Stat().MaxQueryNodeDistance() <<
//          ", scb " << node->Stat().SecondClosestBound() << ".\n";
//    }

//    if (node->Stat().SecondClosestBound() == DBL_MAX)
//   {
//      Log::Warn << "r" << node->Begin() << "c" << node->Count() << " never had "
//          << "the second bound updated.\n";
//    }

  }
  else
  {
    // This node did not have a single owner, but did have a closest query
    // node.  So we will simply loosen that bound.  The loosening here is too
    // loose; TODO: tighten to the max cluster movement in the closest query
    // node.
    if (node->Stat().MaxQueryNodeDistance() != DBL_MAX)
      node->Stat().MaxQueryNodeDistance() += clusterDistances[clusters];
    if (node->Stat().MinQueryNodeDistance() != DBL_MAX)
      node->Stat().MinQueryNodeDistance() += clusterDistances[clusters];

    // Since the node didn't have an owner, it can't be Hamerly pruned.
    node->Stat().HamerlyPruned() = false;
    node->Stat().Owner() = centroids.n_cols;
  }

  node->Stat().Iteration() = iteration;
  node->Stat().ClustersPruned() = (node->Parent() == NULL) ? 0 : -1;
  // We have to set the closest query node to NULL because the cluster tree will
  // be rebuilt.
  node->Stat().ClosestQueryNode() = NULL;

//  if (node->Begin() == 16954)
//    Log::Warn << "scb for r16954c" << node->Count() << " updated to " <<
//node->Stat().SecondClosestBound() << ".\n";

//  if (!node->Stat().HamerlyPruned())
    for (size_t i = 0; i < node->NumChildren(); ++i)
      TreeUpdate(&node->Child(i), clusters, clusterDistances, assignments,
          centroids, dataset, oldFromNew, hamerlyPruned);

  node->Stat().LastSecondClosestBound() = node->Stat().SecondClosestBound() -
      clusterDistances[clusters];
  // This should change later, but I'm not yet sure how to do it.
  node->Stat().SecondClosestBound() = DBL_MAX;
  node->Stat().SecondClosestQueryNode() = NULL;

  if (node->Parent() == NULL)
    Log::Info << "Total Hamerly pruned points: " << hamerlyPruned << ".\n";
}

} // namespace kmeans
} // namespace mlpack

#endif
