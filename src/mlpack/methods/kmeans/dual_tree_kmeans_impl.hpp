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

  // Now run the dual-tree algorithm.
  typedef DualTreeKMeansRules<MetricType, TreeType> RulesType;
  RulesType rules(dataset, centroids, newCentroids, counts, oldFromNewCentroids,
      iteration, clusterDistances, distances, assignments, distanceIteration,
      metric);

  // Use the dual-tree traverser.
//typename TreeType::template DualTreeTraverser<RulesType> traverser(rules);
  typename TreeType::template BreadthFirstDualTreeTraverser<RulesType>
      traverser(rules);

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
  Log::Info << clusterDistances.t();

  delete centroidTree;

  ++iteration;
  return std::sqrt(residual);
}

/*
template<typename MetricType, typename MatType, typename TreeType>
void DualTreeKMeans<MetricType, MatType, TreeType>::ClusterTreeUpdate(
    TreeType* node)
{
  // We will abuse stat.owner to hold the cluster with the most change.
  // stat.minQueryNodeDistance will hold the distance.
  double maxChange = 0.0;
  size_t maxChangeCluster = 0;

  for (size_t i = 0; i < node->NumChildren(); ++i)
  {
    ClusterTreeUpdate(&node->Child(i));

    const double nodeChange = node->Child(i).Stat().MinQueryNodeDistance();
    if (nodeChange > maxChange)
    {
      maxChange = nodeChange;
      maxChangeCluster = node->Child(i).Stat().Owner();
    }
  }

  for (size_t i = 0; i < node->NumPoints(); ++i)
  {
    const size_t cluster = oldFromNewCentroids[node->Point(i)];
    const double pointChange = clusterDistances[cluster];
    if (pointChange > maxChange)
    {
      maxChange = pointChange;
      maxChangeCluster = cluster;
    }
  }

  node->Stat().Owner() = maxChangeCluster;
  node->Stat().MinQueryNodeDistance() = maxChange;
}

template<typename MetricType, typename MatType, typename TreeType>
void DualTreeKMeans<MetricType, MatType, TreeType>::TreeUpdate(
    TreeType* node)
{
  // This is basically IterationUpdate(), but pulled out to be separate from the
  // actual dual-tree algorithm.

  // First, update the iteration.
  const size_t itDiff = node->Stat().Iteration() - iteration;

  if (itDiff == 1)
  {
    // The easy case.
    if (node->Stat().Owner() < centroids.n_cols)
    {
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
          nonOwner->Stat().MinQueryNodeDistance() /* abused from earlier *;
      if (closestUpperBound <= tightestLowerBound)
      {
        // Then the owner must not have changed.

      }
    }
  }
}
*/


} // namespace kmeans
} // namespace mlpack

#endif
