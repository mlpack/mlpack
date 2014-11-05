/**
 * @file dual_tree_kmeans_statistic.hpp
 * @author Ryan Curtin
 *
 * Statistic for dual-tree k-means traversal.
 */
#ifndef __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_STATISTIC_HPP
#define __MLPACK_METHODS_KMEANS_DUAL_TREE_KMEANS_STATISTIC_HPP

namespace mlpack {
namespace kmeans {

class DualTreeKMeansStatistic
{
 public:
  DualTreeKMeansStatistic() { /* Nothing to do. */ }

  template<typename TreeType>
  DualTreeKMeansStatistic(TreeType& node) :
      closestQueryNode(NULL),
      minQueryNodeDistance(DBL_MAX),
      maxQueryNodeDistance(DBL_MAX),
      clustersPruned(0),
      iteration(size_t() - 1)
  {
    // Empirically calculate the centroid.
    centroid.zeros(node.Dataset().n_rows);
    for (size_t i = 0; i < node.NumPoints(); ++i)
      centroid += node.Dataset().col(node.Point(i));

    for (size_t i = 0; i < node.NumChildren(); ++i)
      centroid += node.Child(i).NumDescendants() *
          node.Child(i).Stat().Centroid();

    centroid /= node.NumDescendants();
  }

  //! Return the centroid.
  const arma::vec& Centroid() const { return centroid; }
  //! Modify the centroid.
  arma::vec& Centroid() { return centroid; }

  //! Get the current closest query node.
  void* ClosestQueryNode() const { return closestQueryNode; }
  //! Modify the current closest query node.
  void*& ClosestQueryNode() { return closestQueryNode; }

  //! Get the minimum distance to the closest query node.
  double MinQueryNodeDistance() const { return minQueryNodeDistance; }
  //! Modify the minimum distance to the closest query node.
  double& MinQueryNodeDistance() { return minQueryNodeDistance; }

  //! Get the maximum distance to the closest query node.
  double MaxQueryNodeDistance() const { return maxQueryNodeDistance; }
  //! Modify the maximum distance to the closest query node.
  double& MaxQueryNodeDistance() { return maxQueryNodeDistance; }

  //! Get the number of clusters that have been pruned during this iteration.
  size_t ClustersPruned() const { return clustersPruned; }
  //! Modify the number of clusters that have been pruned during this iteration.
  size_t& ClustersPruned() { return clustersPruned; }

  //! Get the current iteration.
  size_t Iteration() const { return iteration; }
  //! Modify the current iteration.
  size_t& Iteration() { return iteration; }

  //! Get the current owner (if any) of these reference points.
  size_t Owner() const { return owner; }
  //! Modify the current owner (if any) of these reference points.
  size_t& Owner() { return owner; }

 private:
  //! The empirically calculated centroid of the node.
  arma::vec centroid;

  //! The current closest query node to this reference node.
  void* closestQueryNode;
  //! The minimum distance to the closest query node.
  double minQueryNodeDistance;
  //! The maximum distance to the closest query node.
  double maxQueryNodeDistance;

  //! The number of clusters that have been pruned.
  size_t clustersPruned;
  //! The current iteration.
  size_t iteration;
  //! The owner of these reference nodes (centroids.n_cols if there is no
  //! owner).
  size_t owner;
};

} // namespace kmeans
} // namespace mlpack

#endif
