/**
 * @file pelleg_moore_kmeans_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of Pelleg-Moore's 'blacklist' algorithm for k-means
 * clustering.
 */
#ifndef __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_IMPL_HPP
#define __MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_IMPL_HPP

#include "pelleg_moore_kmeans.hpp"
#include "pelleg_moore_kmeans_rules.hpp"

namespace mlpack {
namespace kmeans {

template<typename MetricType, typename MatType>
PellegMooreKMeans<MetricType, MatType>::PellegMooreKMeans(
    const MatType& dataset,
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

template<typename MetricType, typename MatType>
PellegMooreKMeans<MetricType, MatType>::~PellegMooreKMeans()
{
  if (tree)
    delete tree;
}

// Run a single iteration.
template<typename MetricType, typename MatType>
double PellegMooreKMeans<MetricType, MatType>::Iterate(
    const arma::mat& centroids,
    arma::mat& newCentroids,
    arma::Col<size_t>& counts)
{
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Create rules object.
  typedef PellegMooreKMeansRules<MetricType, TreeType> RulesType;
  RulesType rules(dataset, centroids, newCentroids, counts, metric);

  // Use single-tree traverser.
  typename TreeType::template SingleTreeTraverser<RulesType> traverser(rules);

  // Now, do a traversal with a fake query index (since the query index is
  // irrelevant; we are checking each node with all clusters.
  traverser.Traverse(0, *tree);

  distanceCalculations += rules.DistanceCalculations();

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts[c] == 0)
    {
      newCentroids.col(c).fill(DBL_MAX); // Should have happened anyway I think.
    }
    else
    {
      newCentroids.col(c) /= counts(c);
      residual += std::pow(metric.Evaluate(centroids.col(c),
                                           newCentroids.col(c)), 2.0);
    }
  }
  distanceCalculations += centroids.n_cols;

  return std::sqrt(residual);
}

} // namespace kmeans
} // namespace mlpack

#endif
