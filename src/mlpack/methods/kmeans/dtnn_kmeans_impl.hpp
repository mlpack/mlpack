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
  return new TreeType(dataset);
}

template<typename MetricType, typename MatType, typename TreeType>
DTNNKMeans<MetricType, MatType, TreeType>::DTNNKMeans(const MatType& dataset,
                                                      MetricType& metric) :
    datasetOrig(dataset),
    dataset(tree::TreeTraits<TreeType>::RearrangesDataset ? datasetCopy :
        datasetOrig),
    metric(metric),
    distanceCalculations(0)
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
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Build a tree on the centroids.
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree = BuildTree<TreeType>(
      const_cast<typename TreeType::Mat&>(centroids), oldFromNewCentroids);

  // We won't use the AllkNN class here because we have our own set of rules.
  // This is a lot of overhead.  We don't need the distances.
  arma::mat distances(5, dataset.n_cols);
  arma::Mat<size_t> assignments(5, dataset.n_cols);
  distances.fill(DBL_MAX);
  assignments.fill(size_t(-1));
  typedef DTNNKMeansRules<MetricType, TreeType> RuleType;
  RuleType rules(centroids, dataset, assignments, distances, metric);

  // Now construct the traverser ourselves.
  typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*tree, *centroidTree);

  distanceCalculations += rules.BaseCases() + rules.Scores();

  // From the assignments, calculate the new centroids and counts.
  for (size_t i = 0; i < dataset.n_cols; ++i)
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

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  double maxMovement = 0.0;
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    // Get the mapping to the old cluster, if necessary.
    const size_t old = (tree::TreeTraits<TreeType>::RearrangesDataset) ?
        oldFromNewCentroids[c] : c;
    if (counts[old] == 0)
    {
      newCentroids.col(old).fill(DBL_MAX);
    }
    else
    {
      newCentroids.col(old) /= counts(old);
      const double movement = metric.Evaluate(centroids.col(c),
          newCentroids.col(old));
      residual += std::pow(movement, 2.0);

      if (movement > maxMovement)
        maxMovement = movement;
    }
  }
  distanceCalculations += centroids.n_cols;

  UpdateTree(*tree, maxMovement);

  delete centroidTree;

  return std::sqrt(residual);
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::UpdateTree(
    TreeType& node,
    const double tolerance)
{
  if (node.Stat().FirstBound() != DBL_MAX)
    node.Stat().FirstBound() += tolerance;
  if (node.Stat().SecondBound() != DBL_MAX)
    node.Stat().SecondBound() += tolerance;
  if (node.Stat().Bound() != DBL_MAX)
    node.Stat().Bound() += tolerance;

  for (size_t i = 0; i < node.NumChildren(); ++i)
    UpdateTree(node.Child(i), tolerance);
}

} // namespace kmeans
} // namespace mlpack

#endif
