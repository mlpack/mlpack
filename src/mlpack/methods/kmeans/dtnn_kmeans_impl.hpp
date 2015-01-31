/**
 * @file dtnn_kmeans_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of a Lloyd iteration which uses dual-tree nearest neighbor
 * search as a black box.  The conditions under which this will perform best are
 * probably limited to the case where k is close to the number of points in the
 * dataset, and the number of iterations of the k-means algorithm will be few.
 */
#ifndef __MLPACK_METHODS_KMEANS_DTNN_KMEANS_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_DTNN_KMEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "dtnn_kmeans.hpp"

#include "dtnn_rules.hpp"

namespace mlpack {
namespace kmeans {

//! Call the tree constructor that does mapping.
template<typename TreeType>
TreeType* BuildTree(
    typename TreeType::Mat& dataset,
    std::vector<size_t>& oldFromNew,
    typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == true, TreeType*
    >::type = 0)
{
  // This is a hack.  I know this will be BinarySpaceTree, so force a leaf size
  // of two.
  return new TreeType(dataset, oldFromNew, 1);
}

//! Call the tree constructor that does not do mapping.
template<typename TreeType>
TreeType* BuildTree(
    const typename TreeType::Mat& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename boost::enable_if_c<
        tree::TreeTraits<TreeType>::RearrangesDataset == false, TreeType*
    >::type = 0)
{
  return new TreeType(dataset, 1);
}

template<typename MetricType, typename MatType, typename TreeType>
DTNNKMeans<MetricType, MatType, TreeType>::DTNNKMeans(const MatType& dataset,
                                                      MetricType& metric) :
    datasetOrig(dataset),
    dataset(tree::TreeTraits<TreeType>::RearrangesDataset ? datasetCopy :
        datasetOrig),
    metric(metric),
    distanceCalculations(0),
    iteration(0)
{
  Timer::Start("tree_building");

  // Copy the dataset, if necessary.
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
    datasetCopy = datasetOrig;

  // Now build the tree.  We don't need any mappings.
  tree = new TreeType(const_cast<typename TreeType::Mat&>(this->dataset));

  Timer::Stop("tree_building");
}

template<typename MetricType, typename MatType, typename TreeType>
DTNNKMeans<MetricType, MatType, TreeType>::~DTNNKMeans()
{
  if (tree)
    delete tree;
}

// Run a single iteration.
template<typename MetricType, typename MatType, typename TreeType>
double DTNNKMeans<MetricType, MatType, TreeType>::Iterate(
    const arma::mat& centroids,
    arma::mat& newCentroids,
    arma::Col<size_t>& counts)
{
  if (iteration == 0)
  {
    prunedCentroids.zeros(centroids.n_rows, centroids.n_cols);
    prunedCounts.zeros(centroids.n_cols);
  }

  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Build a tree on the centroids.
  arma::mat oldCentroids(centroids); // Slow. :(
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree = BuildTree<TreeType>(
      const_cast<typename TreeType::Mat&>(centroids), oldFromNewCentroids);

  // We won't use the AllkNN class here because we have our own set of rules.
  // This is a lot of overhead.  We don't need the distances.
  arma::mat distances(2, dataset.n_cols);
  arma::Mat<size_t> assignments(2, dataset.n_cols);
  distances.fill(DBL_MAX);
  assignments.fill(size_t(-1));
  typedef DTNNKMeansRules<MetricType, TreeType> RuleType;
  RuleType rules(centroids, dataset, assignments, distances, metric);

  // Now construct the traverser ourselves.
  typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*tree, *centroidTree);

  Log::Info << "This iteration: " << rules.BaseCases() << " base cases, " <<
      rules.Scores() << " scores.\n";
  distanceCalculations += rules.BaseCases() + rules.Scores();

  // From the assignments, calculate the new centroids and counts.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (assignments(0, i) != size_t(-1))
    {
      if (tree::TreeTraits<TreeType>::RearrangesDataset)
      {
        newCentroids.col(oldFromNewCentroids[assignments(0, i)]) +=
            dataset.col(i);
        ++counts(oldFromNewCentroids[assignments(0, i)]);
      }
      else
      {
        newCentroids.col(assignments(0, i)) += dataset.col(i);
        ++counts(assignments(0, i));
      }
    }
  }

  newCentroids += prunedCentroids;
  counts += prunedCounts;

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  double maxMovement = 0.0;
  arma::vec clusterDistances(centroids.n_cols + 1);
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    // Get the mapping to the old cluster, if necessary.
    const size_t old = (tree::TreeTraits<TreeType>::RearrangesDataset) ?
        oldFromNewCentroids[c] : c;
    if (counts[old] == 0)
    {
      newCentroids.col(old).fill(DBL_MAX);
      clusterDistances[old] = 0;
    }
    else
    {
      newCentroids.col(old) /= counts(old);
      const double movement = metric.Evaluate(centroids.col(c),
          newCentroids.col(old));
      clusterDistances[old] = movement;
      residual += std::pow(movement, 2.0);

      if (movement > maxMovement)
        maxMovement = movement;
    }
  }
  clusterDistances[centroids.n_cols] = maxMovement;
  Log::Warn << clusterDistances.t();
  distanceCalculations += centroids.n_cols;

  UpdateTree(*tree, maxMovement, oldCentroids, assignments, distances,
      clusterDistances, oldFromNewCentroids);

  // Reset centroids and counts for things we will collect during pruning.
  prunedCentroids.zeros(centroids.n_rows, centroids.n_cols);
  prunedCounts.zeros(centroids.n_cols);
  PrecalculateCentroids(*tree);

  delete centroidTree;

  ++iteration;

  return std::sqrt(residual);
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::UpdateTree(
    TreeType& node,
    const double tolerance,
    const arma::mat& centroids,
    const arma::Mat<size_t>& assignments,
    const arma::mat& distances,
    const arma::mat& clusterDistances,
    const std::vector<size_t>& oldFromNewCentroids)
{
  // Update iteration.
//  node.Stat().Iteration() = iteration;

  if (node.Stat().Owner() == size_t(-1))
    node.Stat().Owner() = centroids.n_cols;

  // Do the tree update in a depth-first manner: leaves first.
  bool childrenPruned = true;
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    UpdateTree(node.Child(i), tolerance, centroids, assignments, distances,
        clusterDistances, oldFromNewCentroids);
    if (!node.Child(i).Stat().Pruned())
      childrenPruned = false; // Not all children are pruned.
  }

  // Does the node have a single owner?
  // It would be nice if we could do this during the traversal.
  bool singleOwner = true;
  size_t owner = centroids.n_cols + 1;
  node.Stat().MaxClusterDistance() = 0.0;
  node.Stat().SecondClusterBound() = DBL_MAX;
  if (!node.Stat().Pruned() && childrenPruned)
  {
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      // Don't forget to map back from the new cluster index.
      if (owner == centroids.n_cols + 1)
        owner = (tree::TreeTraits<TreeType>::RearrangesDataset) ?
            oldFromNewCentroids[assignments(0, node.Point(i))] :
            oldFromNewCentroids[assignments(0, node.Point(i))];
      else if (owner != oldFromNewCentroids[assignments(0, node.Point(i))])
        singleOwner = false;

      // Update maximum cluster distance and second cluster bound.
      if (distances(0, node.Point(i)) > node.Stat().MaxClusterDistance())
        node.Stat().MaxClusterDistance() = distances(0, node.Point(i));
      if (distances(1, node.Point(i)) < node.Stat().SecondClusterBound())
        node.Stat().SecondClusterBound() = distances(1, node.Point(i));
    }

    for (size_t i = 0; i < node.NumChildren(); ++i)
    {
      if (owner == centroids.n_cols + 1)
        owner = node.Child(i).Stat().Owner();
      else if (node.Child(i).Stat().Owner() == centroids.n_cols)
        singleOwner = false;
      else if (owner != node.Child(i).Stat().Owner())
        singleOwner = false;

      // Update maximum cluster distance and second cluster bound.
      if (node.Child(i).Stat().MaxClusterDistance() >
          node.Stat().MaxClusterDistance())
        node.Stat().MaxClusterDistance() =
            node.Child(i).Stat().MaxClusterDistance();
      if (node.Child(i).Stat().SecondClusterBound() <
          node.Stat().SecondClusterBound())
        node.Stat().SecondClusterBound() =
            node.Child(i).Stat().SecondClusterBound();
    }

    // Okay, now we know if it's owned or not, and by which cluster.
    if (singleOwner)
    {
      node.Stat().Owner() = owner;

      // Sanity check: ensure the owner is right.
      for (size_t i = 0; i < node.NumPoints(); ++i)
      {
        const double ownerDist = metric.Evaluate(dataset.col(node.Point(i)),
            centroids.col(owner));
        for (size_t j = 0; j < centroids.n_cols; ++j)
        {
          const double dist = metric.Evaluate(dataset.col(node.Point(i)),
              centroids.col(j));
          if (dist < ownerDist)
          {
            Log::Warn << node << "...\n" << *node.Parent();
            Log::Fatal << "Point " << node.Point(i) << " was assigned to owner "
                << owner << " but has true owner " << j << "! [" <<
oldFromNewCentroids[assignments(0, node.Point(i))] << " -- " <<
metric.Evaluate(dataset.col(node.Point(i)),
centroids.col(oldFromNewCentroids[assignments(0, node.Point(i))])) << "] " <<
distances(0, node.Point(i)) << " " <<
oldFromNewCentroids[assignments(0, node.Point(i))] << " " <<
oldFromNewCentroids[assignments(0, node.Point(i - 1))] << ".\n";
          }
        }
      }

      // What is the maximum distance to the closest cluster in the node?
      if (node.Stat().MaxClusterDistance() +
          clusterDistances[node.Stat().Owner()] <
          node.Stat().SecondClusterBound() - clusterDistances[centroids.n_cols])
      {
        node.Stat().Pruned() = true;
      }
    }
  }
  else if (node.Stat().Pruned())
  {
    // The node was pruned last iteration.  See if the node can remain pruned.
    singleOwner = false;

    node.Stat().Pruned() = false;
    node.Stat().FirstBound() = DBL_MAX;
    node.Stat().SecondBound() = DBL_MAX;
    node.Stat().Bound() = DBL_MAX;
  }
  else
  {
    // The children haven't been pruned, so we can't.
    // This node was not pruned last iteration, so we simply need to adjust the
    // bounds.
    if (node.Stat().FirstBound() != DBL_MAX)
      node.Stat().FirstBound() += tolerance;
    if (node.Stat().SecondBound() != DBL_MAX)
      node.Stat().SecondBound() += tolerance;
    if (node.Stat().Bound() != DBL_MAX)
      node.Stat().Bound() += tolerance;
  }
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::PrecalculateCentroids(
    TreeType& node)
{
  if (node.Stat().Pruned() && node.Stat().Owner() < prunedCentroids.n_cols)
  {
    prunedCentroids.col(node.Stat().Owner()) += node.Stat().Centroid() *
        node.NumDescendants();
    prunedCounts(node.Stat().Owner()) += node.NumDescendants();
  }
  else
  {
    for (size_t i = 0; i < node.NumChildren(); ++i)
      PrecalculateCentroids(node.Child(i));
  }
}

} // namespace kmeans
} // namespace mlpack

#endif
