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

namespace mlpack {
namespace kmeans {

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
  arma::mat centroidsCopy;

  // Build a tree on the centroids.
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree;
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    // Manually set leaf size of 2.  This may not always be appropriate.
    centroidsCopy = centroids;
    centroidTree = new TreeType(centroidsCopy, oldFromNewCentroids, 2);
  }
  else
  {
    centroidTree = new TreeType(centroidsCopy);
  }

  typedef neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType,
      TreeType> AllkNNType;
  AllkNNType allknn(centroidTree, tree,
      (tree::TreeTraits<TreeType>::RearrangesDataset) ? centroidsCopy :
      centroids, dataset, false, metric);

  // This is a lot of overhead.  We don't need the distances.
  arma::mat distances;
  arma::Mat<size_t> assignments;
  allknn.Search(1, assignments, distances);
  distanceCalculations += allknn.BaseCases() + allknn.Scores();

  // From the assignments, calculate the new centroids and counts.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (tree::TreeTraits<TreeType>::RearrangesDataset)
    {
      newCentroids.col(oldFromNewCentroids[assignments[i]]) += dataset.col(i);
      ++counts(oldFromNewCentroids[assignments[i]]);
    }
    else
    {
      newCentroids.col(assignments[i]) += dataset.col(i);
      ++counts(i);
    }
  }

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  double maxMovement = 0.0;
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts[c] == 0)
    {
      newCentroids.col(c).fill(DBL_MAX); // Should have happened anyway I think.
    }
    else
    {
      newCentroids.col(c) /= counts(c);
      const double movement = metric.Evaluate(centroids.col(c),
          newCentroids.col(c));
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
