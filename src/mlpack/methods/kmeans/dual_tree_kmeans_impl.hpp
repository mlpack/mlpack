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
  tree = new TreeType(const_cast<typename TreeType::Mat&>(this->dataset), 1);

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
  else if (&potentialChild == &potentialParent)
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
  if (node->Begin() == 26038)
    Log::Warn << "r26038c" << node->Count() << " has owner " <<
node->Stat().Owner() << ".\n";
  if (node->Parent() != NULL && node->Parent()->Stat().Owner() < clusters)
    node->Stat().Owner() = node->Parent()->Stat().Owner();
  if (node->Begin() == 26038)
    Log::Warn << "r26038c" << node->Count() << " has owner " <<
node->Stat().Owner() << " after parent check.\n";

  const size_t cluster = assignments[node->Descendant(0)];
  bool allSame = true;
  for (size_t i = 1; i < node->NumDescendants(); ++i)
  {
    if (assignments[node->Descendant(i)] != cluster)
    {
      allSame = false;
      break;
    }
  }

  if (allSame)
    node->Stat().Owner() = cluster;
  else
    node->Stat().Owner() = centroids.n_cols;
  if (node->Begin() == 26038)
    Log::Warn << "r26038c" << node->Count() << " has manually set owner " <<
node->Stat().Owner() << ".\n";

  const bool prunedLastIteration = node->Stat().HamerlyPruned();
  node->Stat().HamerlyPruned() = false;

  if (node->Begin() == 26038)
    Log::Warn << "r26038c" << node->Count() << " has owner " <<
node->Stat().Owner() << ".\n";

  // The easy case: this node had an owner.
  if (node->Stat().Owner() < clusters)
  {
    // Verify correctness...
    for (size_t i = 0; i < node->NumDescendants(); ++i)
    {
      size_t closest = clusters;
      double closestDistance = DBL_MAX;
      arma::vec distances(centroids.n_cols);
      for (size_t j = 0; j < centroids.n_cols; ++j)
      {
        const double distance = metric.Evaluate(centroids.col(j),
            dataset.col(node->Descendant(i)));
        if (distance < closestDistance)
        {
          closest = j;
          closestDistance = distance;
        }
        distances(j) = distance;
      }

      if (closest != node->Stat().Owner())
      {
        Log::Warn << distances.t();
        Log::Fatal << "Point " << node->Descendant(i) << " mistakenly assigned "
            << "to cluster " << node->Stat().Owner() << ", but should be " <<
closest << "!  It's part of node r" << node->Begin() << "c" << node->Count() <<
".\n";
      }
    }

    // During the last iteration, this node was pruned.
    const size_t owner = node->Stat().Owner();
    if (node->Stat().MaxQueryNodeDistance() != DBL_MAX)
      node->Stat().MaxQueryNodeDistance() += clusterDistances[owner];
    if (node->Stat().MinQueryNodeDistance() != DBL_MAX)
      node->Stat().MinQueryNodeDistance() += clusterDistances[owner];

    if (prunedLastIteration)
    {
      // Can we continue being Hamerly pruned?  If not, we'll have to update the
      // bound next iteration.
      if (node->Begin() == 26038)
        Log::Warn << "r26038c" << node->Count() << ": check sustained Hamerly "
            << "prune with MQND " << node->Stat().MaxQueryNodeDistance() << ", "
            << "lscb " << node->Stat().LastSecondClosestBound() << ", cd "
            << clusterDistances[clusters] << ".\n";
      if (node->Stat().MaxQueryNodeDistance() <
          node->Stat().LastSecondClosestBound() - clusterDistances[clusters])
      {
        node->Stat().HamerlyPruned() = true;
        if (!node->Parent()->Stat().HamerlyPruned())
          hamerlyPruned += node->NumDescendants();
      }
    }
    else
    {
      if (node->Begin() == 26038)
      {
        if (node->Stat().ClosestQueryNode() != NULL)
          Log::Warn << "r26038c" << node->Count() << " CQN: " << ((TreeType*)
  node->Stat().ClosestQueryNode())->Begin() << "c" << ((TreeType*)
  node->Stat().ClosestQueryNode())->Count() << ".\n";
        if (node->Stat().SecondClosestQueryNode() != NULL)
          Log::Warn << "r26038c" << node->Count() << " SCQN: " << ((TreeType*)
  node->Stat().SecondClosestQueryNode())->Begin() << "c" << ((TreeType*)
  node->Stat().SecondClosestQueryNode())->Count() << ".\n";
        Log::Warn << "Attempt hamerly prune r26038c" << node->Count() << " with "
            << "MQND " << node->Stat().MaxQueryNodeDistance() << " and smqnd "
            << node->Stat().SecondMinQueryNodeDistance() << " and cluster d "
            << clusterDistances[clusters] << ".\n";
      }

      // Now we check for a Hamerly prune.  We know that we have an accurate
      // second bound since nothing can be pruned.
      if (node->Stat().MaxQueryNodeDistance() /* already adjusted */ <
          node->Stat().SecondMinQueryNodeDistance() - clusterDistances[clusters])
      {
        node->Stat().HamerlyPruned() = true;
        if (!node->Parent()->Stat().HamerlyPruned())
          hamerlyPruned += node->NumDescendants();
      }
    }
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

  bool allPruned = true;
  size_t owner = clusters;
  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    TreeUpdate(&node->Child(i), clusters, clusterDistances, assignments,
        centroids, dataset, oldFromNew, hamerlyPruned);
    if (!node->Child(i).Stat().HamerlyPruned())
      allPruned = false;
    else if (owner == clusters)
      owner = node->Child(i).Stat().Owner();
    else if (owner < clusters && owner != node->Child(i).Stat().Owner())
      owner = clusters + 1;
  }

  if (node->NumChildren() == 0 && !node->Stat().HamerlyPruned())
    allPruned = false;

  if (allPruned && owner < clusters && !node->Stat().HamerlyPruned())
  {
    if (node->Begin() == 26038)
      Log::Warn << "Set r" << node->Begin() << "c" << node->Count() << " to be "
          << "Hamerly pruned.\n";
    node->Stat().HamerlyPruned() = true;
  }

  if (node->Begin() == 26038 && node->Stat().HamerlyPruned())
    Log::Warn << "r" << node->Begin() << "c" << node->Count() << " is Hamerly "
        << "pruned.\n";

  node->Stat().Iteration() = iteration;
  node->Stat().ClustersPruned() = (node->Parent() == NULL) ? 0 : -1;
  // We have to set the closest query node to NULL because the cluster tree will
  // be rebuilt.
  node->Stat().ClosestQueryNode() = NULL;

  if (prunedLastIteration)
    node->Stat().LastSecondClosestBound() -= clusterDistances[clusters];
  else
    node->Stat().LastSecondClosestBound() =
        node->Stat().SecondMinQueryNodeDistance() - clusterDistances[clusters];
  node->Stat().MinQueryNodeDistance() = DBL_MAX;
  if (prunedLastIteration && !node->Stat().HamerlyPruned())
    node->Stat().MaxQueryNodeDistance() = DBL_MAX;
  node->Stat().SecondMinQueryNodeDistance() = DBL_MAX;
  node->Stat().SecondMaxQueryNodeDistance() = DBL_MAX;
  // This should change later, but I'm not yet sure how to do it.
//  node->Stat().SecondClosestBound() = DBL_MAX;
//  node->Stat().SecondClosestQueryNode() = NULL;

  if (node->Parent() == NULL)
    Log::Info << "Total Hamerly pruned points: " << hamerlyPruned << ".\n";
}

} // namespace kmeans
} // namespace mlpack

#endif
