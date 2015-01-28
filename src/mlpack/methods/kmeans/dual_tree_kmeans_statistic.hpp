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
      secondClosestQueryNode(NULL),
      minQueryNodeDistance(DBL_MAX),
      maxQueryNodeDistance(DBL_MAX),
      secondMinQueryNodeDistance(DBL_MAX),
      secondMaxQueryNodeDistance(DBL_MAX),
      lastSecondClosestBound(DBL_MAX),
      hamerlyPruned(false),
      clustersPruned(size_t(-1)),
      iteration(size_t() - 1),
      owner(size_t(-1)),
      firstBound(DBL_MAX),
      secondBound(DBL_MAX),
      bound(DBL_MAX),
      lastDistanceNode(NULL),
      lastDistance(0.0)
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

  //! Get the second closest query node.
  void* SecondClosestQueryNode() const { return secondClosestQueryNode; }
  //! Modify the second closest query node.
  void*& SecondClosestQueryNode() { return secondClosestQueryNode; }

  //! Get the minimum distance to the closest query node.
  double MinQueryNodeDistance() const { return minQueryNodeDistance; }
  //! Modify the minimum distance to the closest query node.
  double& MinQueryNodeDistance() { return minQueryNodeDistance; }

  //! Get the maximum distance to the closest query node.
  double MaxQueryNodeDistance() const { return maxQueryNodeDistance; }
  //! Modify the maximum distance to the closest query node.
  double& MaxQueryNodeDistance() { return maxQueryNodeDistance; }

  //! Get the minimum distance to the second closest query node.
  double SecondMinQueryNodeDistance() const
  { return secondMinQueryNodeDistance; }
  //! Modify the minimum distance to the second closest query node.
  double& SecondMinQueryNodeDistance() { return secondMinQueryNodeDistance; }

  //! Get the maximum distance to the second closest query node.
  double SecondMaxQueryNodeDistance() const
  { return secondMaxQueryNodeDistance; }
  //! Modify the maximum distance to the second closest query node.
  double& SecondMaxQueryNodeDistance() { return secondMaxQueryNodeDistance; }

  //! Get last iteration's second closest bound.
  double LastSecondClosestBound() const { return lastSecondClosestBound; }
  //! Modify last iteration's second closest bound.
  double& LastSecondClosestBound() { return lastSecondClosestBound; }

  //! Get whether or not this node is Hamerly pruned this iteration.
  bool HamerlyPruned() const { return hamerlyPruned; }
  //! Modify whether or not this node is Hamerly pruned this iteration.
  bool& HamerlyPruned() { return hamerlyPruned; }

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

  // For nearest neighbor search.

  //! Get the first bound.
  double FirstBound() const { return firstBound; }
  //! Modify the first bound.
  double& FirstBound() { return firstBound; }
  //! Get the second bound.
  double SecondBound() const { return secondBound; }
  //! Modify the second bound.
  double& SecondBound() { return secondBound; }
  //! Get the overall bound.
  double Bound() const { return bound; }
  //! Modify the overall bound.
  double& Bound() { return bound; }
  //! Get the last distance evaluation node.
  void* LastDistanceNode() const { return lastDistanceNode; }
  //! Modify the last distance evaluation node.
  void*& LastDistanceNode() { return lastDistanceNode; }
  //! Get the last distance calculation.
  double LastDistance() const { return lastDistance; }
  //! Modify the last distance calculation.
  double& LastDistance() { return lastDistance; }

 private:
  //! The empirically calculated centroid of the node.
  arma::vec centroid;

  //! The current closest query node to this reference node.
  void* closestQueryNode;
  //! The second closest query node.
  void* secondClosestQueryNode;
  //! The minimum distance to the closest query node.
  double minQueryNodeDistance;
  //! The maximum distance to the closest query node.
  double maxQueryNodeDistance;
  //! The minimum distance to the second closest query node.
  double secondMinQueryNodeDistance;
  //! The maximum distance to the second closest query node.
  double secondMaxQueryNodeDistance;

  //! The second closest lower bound, on the previous iteration.
  double lastSecondClosestBound;
  //! Whether or not this node is pruned for the next iteration.
  bool hamerlyPruned;

  //! The number of clusters that have been pruned.
  size_t clustersPruned;
  //! The current iteration.
  size_t iteration;
  //! The owner of these reference nodes (centroids.n_cols if there is no
  //! owner).
  size_t owner;

  // For nearest neighbor search.

  double firstBound;
  double secondBound;
  double bound;
  void* lastDistanceNode;
  double lastDistance;
};

} // namespace kmeans
} // namespace mlpack

#endif
