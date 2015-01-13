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
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree = BuildTree<TreeType>(
      const_cast<typename TreeType::Mat&>(centroids), oldFromNewCentroids);

  // Now calculate distances between centroids.
  neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType, TreeType>
      nns(centroidTree, centroids);
  arma::mat interclusterDistances;
  arma::Mat<size_t> closestClusters; // We don't actually care about these.
  nns.Search(1, closestClusters, interclusterDistances);

  distanceCalculations += nns.BaseCases();
  distanceCalculations += nns.Scores();

  // Update FirstBound().
  ClusterTreeUpdate(centroidTree);

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
//  Log::Info << clusterDistances.t();

  // Update the tree with the centroid movement information.
  TreeUpdate(tree, centroids.n_cols, clusterDistances);

  delete centroidTree;

  ++iteration;
  return std::sqrt(residual);
}

template<typename MetricType, typename MatType, typename TreeType>
void DualTreeKMeans<MetricType, MatType, TreeType>::ClusterTreeUpdate(
    TreeType* node)
{
  // Just update the first bound, after recursing to the bottom.
  double firstBound = 0.0;
  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    ClusterTreeUpdate(&node->Child(i));
    if (node->Child(i).Stat().FirstBound() >= firstBound)
      firstBound = node->Child(i).Stat().FirstBound();
  }

  node->Stat().FirstBound() = firstBound;
}

template<typename MetricType, typename MatType, typename TreeType>
void DualTreeKMeans<MetricType, MatType, TreeType>::TreeUpdate(
    TreeType* node,
    const size_t clusters,
    const arma::vec& clusterDistances)
{
  // This is basically IterationUpdate(), but pulled out to be separate from the
  // actual dual-tree algorithm.

  if (node->Parent() != NULL && node->Parent()->Stat().Owner() < clusters)
    node->Stat().Owner() = node->Parent()->Stat().Owner();

  // The easy case: this node had an owner.
  if (node->Stat().Owner() < clusters)
  {
    // During the last iteration, this node was pruned.
    const size_t owner = node->Stat().Owner();
    if (node->Stat().MaxQueryNodeDistance() != DBL_MAX)
      node->Stat().MaxQueryNodeDistance() += clusterDistances[owner];
    if (node->Stat().MinQueryNodeDistance() != DBL_MAX)
      node->Stat().MinQueryNodeDistance() += clusterDistances[owner];

/*
    // During the last iteration, this node was pruned.  In addition, we have
    // cached a lower bound on the second closest cluster.  So, use the
    // triangle inequality: if the maximum distance between the point and the
    // cluster centroid plus the distance that centroid moved is less than the
    // lower bound minus the maximum moving centroid, then this cluster *must*
    // still have the same owner.
    const size_t owner = node->Stat().Owner();
    const double closestUpperBound = node->Stat().MaxQueryNodeDistance() +
        clusterDistances[owner];
    const TreeType* nonOwner = (TreeType*) node->Stat().ClosestNonOwner();
    const double tightestLowerBound = node->Stat().ClosestNonOwnerDistance() -
        nonOwner->Stat().MinQueryNodeDistance();
    if (closestUpperBound <= tightestLowerBound)
    {
      // Then the owner must not have changed.
    }
*/
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
  }

  node->Stat().Iteration() = iteration;
  node->Stat().ClustersPruned() = (node->Parent() == NULL) ? 0 : -1;
  // We have to set the closest query node to NULL because the cluster tree will
  // be rebuilt.
  node->Stat().ClosestQueryNode() = NULL;
//  node->Stat().MaxQueryNodeDistance() = DBL_MAX;
//  node->Stat().MinQueryNodeDistance() = DBL_MAX;

  for (size_t i = 0; i < node->NumChildren(); ++i)
    TreeUpdate(&node->Child(i), clusters, clusterDistances);
}


} // namespace kmeans
} // namespace mlpack

#endif
