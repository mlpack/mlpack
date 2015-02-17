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
  return new TreeType(dataset, oldFromNew, 2);
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
  return new TreeType(dataset);
}

template<typename MetricType, typename MatType, typename TreeType>
DTNNKMeans<MetricType, MatType, TreeType>::DTNNKMeans(const MatType& dataset,
                                                      MetricType& metric) :
    datasetOrig(dataset),
    dataset(tree::TreeTraits<TreeType>::RearrangesDataset ? datasetCopy :
        datasetOrig),
    metric(metric),
    distanceCalculations(0),
    iteration(0),
    upperBounds(dataset.n_cols),
    lowerBounds(dataset.n_cols),
    prunedPoints(dataset.n_cols, false), // Fill with false.
    assignments(dataset.n_cols),
    visited(dataset.n_cols, false) // Fill with false.
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
  // Reset information.
  upperBounds.fill(DBL_MAX);
  lowerBounds.fill(DBL_MAX);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    prunedPoints[i] = false;
    visited[i] = false;
  }

  // Build a tree on the centroids.
  arma::mat oldCentroids(centroids); // Slow. :(
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree = BuildTree<TreeType>(
      const_cast<typename TreeType::Mat&>(centroids), oldFromNewCentroids);

/*
  Timer::Start("knn");
  // Find the nearest neighbors of each of the clusters.
  neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType, TreeType>
      nns(centroidTree, centroids);
  arma::mat interclusterDistances;
  arma::Mat<size_t> closestClusters; // We don't actually care about these.
  nns.Search(1, closestClusters, interclusterDistances);
  distanceCalculations += nns.BaseCases() + nns.Scores();
  Timer::Stop("knn");
*/

  // We won't use the AllkNN class here because we have our own set of rules.
  typedef DTNNKMeansRules<MetricType, TreeType> RuleType;
  RuleType rules(centroids, dataset, assignments, upperBounds, lowerBounds,
      metric, prunedPoints, oldFromNewCentroids, visited);

  typename TreeType::template BreadthFirstDualTreeTraverser<RuleType>
      traverser(rules);

  // Set the number of pruned centroids in the root to 0.
  tree->Stat().Pruned() = 0;
  traverser.Traverse(*tree, *centroidTree);
  distanceCalculations += rules.BaseCases() + rules.Scores();

  // Now we need to extract the clusters.
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);
  ExtractCentroids(*tree, newCentroids, counts, oldFromNewCentroids,
      oldCentroids);

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  arma::vec clusterDistances(centroids.n_cols + 1);
  clusterDistances[centroids.n_cols] = 0.0;
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

      if (movement > clusterDistances[centroids.n_cols])
        clusterDistances[centroids.n_cols] = movement;
    }
  }
  distanceCalculations += centroids.n_cols;

  UpdateTree(*tree, clusterDistances, oldFromNewCentroids, newCentroids);

  delete centroidTree;

  ++iteration;

  return std::sqrt(residual);
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::UpdateTree(
    TreeType& node,
    arma::vec& clusterDistances,
    std::vector<size_t>& oldFromNewCentroids,
    arma::mat& newCentroids)
{
  node.Stat().StaticPruned() = false;

  if ((node.Stat().Pruned() == clusterDistances.n_elem - 1) &&
      (node.Stat().Owner() < clusterDistances.n_elem - 1))
  {
    // Adjust bounds.
    node.Stat().UpperBound() += clusterDistances[node.Stat().Owner()];
    node.Stat().LowerBound() -= clusterDistances[clusterDistances.n_elem - 1];
    if (node.Stat().UpperBound() < node.Stat().LowerBound())
    {
      node.Stat().StaticPruned() = true;
    }
    else
    {
      // Tighten bound.
      node.Stat().UpperBound() =
          node.MaxDistance(newCentroids.col(node.Stat().Owner()));
      ++distanceCalculations;
      if (node.Stat().UpperBound() < node.Stat().LowerBound())
      {
        node.Stat().StaticPruned() = true;
      }
    }
  }

  if (!node.Stat().StaticPruned())
  {
    node.Stat().UpperBound() = DBL_MAX;
    node.Stat().LowerBound() = DBL_MAX;
    node.Stat().Pruned() = size_t(-1);
    node.Stat().Owner() = clusterDistances.n_elem - 1;
    node.Stat().StaticPruned() = false;
  }

  for (size_t i = 0; i < node.NumChildren(); ++i)
    UpdateTree(node.Child(i), clusterDistances, oldFromNewCentroids,
        newCentroids);
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::ExtractCentroids(
    TreeType& node,
    arma::mat& newCentroids,
    arma::Col<size_t>& newCounts,
    std::vector<size_t>& oldFromNewCentroids,
    arma::mat& centroids)
{
  // Does this node own points?
  if ((node.Stat().Pruned() == newCentroids.n_cols) ||
      (node.Stat().StaticPruned() && node.Stat().Owner() < newCentroids.n_cols))
  {
    const size_t owner = node.Stat().Owner();
    newCentroids.col(owner) += node.Stat().Centroid() * node.NumDescendants();
    newCounts[owner] += node.NumDescendants();

    // Perform the sanity check here.
/*
    for (size_t i = 0; i < node.NumDescendants(); ++i)
    {
      const size_t index = node.Descendant(i);
      arma::vec trueDistances(centroids.n_cols);
      for (size_t j = 0; j < centroids.n_cols; ++j)
      {
        const double dist = metric.Evaluate(dataset.col(index),
                                            centroids.col(j));
        trueDistances[j] = dist;
      }

      arma::uword minIndex;
      const double minDist = trueDistances.min(minIndex);
      if (size_t(minIndex) != owner)
      {
        Log::Warn << trueDistances.t();
        Log::Fatal << "Point " << index << " of node " << node.Point(0) << "c"
<< node.NumDescendants() << " has true minimum cluster " << minIndex << " with "
      << "distance " << minDist << " but node is pruned with upper bound " <<
node.Stat().UpperBound() << " and owner " << node.Stat().Owner() << ".\n";
      }
    }
*/
  }
  else
  {
    // Check each point held in the node.
    // Only check at leaves.
    if (node.NumChildren() == 0)
    {
      for (size_t i = 0; i < node.NumPoints(); ++i)
      {
        const size_t owner = assignments[node.Point(i)];
        newCentroids.col(owner) += dataset.col(node.Point(i));
        ++newCounts[owner];
      }
    }

    // The node is not entirely owned by a cluster.  Recurse.
    for (size_t i = 0; i < node.NumChildren(); ++i)
      ExtractCentroids(node.Child(i), newCentroids, newCounts,
          oldFromNewCentroids, centroids);
  }
}

} // namespace kmeans
} // namespace mlpack

#endif
