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
  return new TreeType(dataset, oldFromNew);
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
    distances(2, dataset.n_cols),
    assignments(2, dataset.n_cols)
{
  prunedPoints.resize(dataset.n_cols, false); // Fill with false.
  lowerSecondBounds.zeros(dataset.n_cols);
  lastOwners.zeros(dataset.n_cols);

  assignments.set_size(2, dataset.n_cols);
  assignments.fill(size_t(-1));
  distances.set_size(2, dataset.n_cols);
  distances.fill(DBL_MAX);

  visited.resize(dataset.n_cols, false);

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
    // The last element stores the maximum.
    clusterDistances.zeros(centroids.n_cols + 1);
  }

  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);

  // Build a tree on the centroids.
  arma::mat oldCentroids(centroids); // Slow. :(
  std::vector<size_t> oldFromNewCentroids;
  TreeType* centroidTree = BuildTree<TreeType>(
      const_cast<typename TreeType::Mat&>(centroids), oldFromNewCentroids);
  // Calculate new from old mappings.
  std::vector<size_t> newFromOldCentroids;
  if (tree::TreeTraits<TreeType>::RearrangesDataset)
  {
    newFromOldCentroids.resize(centroids.n_cols);
    for (size_t i = 0; i < centroids.n_cols; ++i)
      newFromOldCentroids[oldFromNewCentroids[i]] = i;
  }

  Timer::Start("knn");
  // Find the nearest neighbors of each of the clusters.
  neighbor::NeighborSearch<neighbor::NearestNeighborSort, MetricType, TreeType>
      nns(centroidTree, centroids);
  arma::mat interclusterDistances;
  arma::Mat<size_t> closestClusters; // We don't actually care about these.
  nns.Search(1, closestClusters, interclusterDistances);
  distanceCalculations += nns.BaseCases() + nns.Scores();
  Timer::Stop("knn");

  if (iteration != 0)
  {
    // Do the tree update for the previous iteration.

    // Reset centroids and counts for things we will collect during pruning.
    Timer::Start("it_update");
    prunedCentroids.zeros(centroids.n_rows, centroids.n_cols);
    prunedCounts.zeros(centroids.n_cols);
    UpdateTree(*tree, oldCentroids, interclusterDistances, newFromOldCentroids);

    PrecalculateCentroids(*tree);
    Timer::Stop("it_update");
  }

  Timer::Start("tree_mod");
  CoalesceTree(*tree);
  Timer::Stop("tree_mod");

  // We won't use the AllkNN class here because we have our own set of rules.
  // This is a lot of overhead.  We don't need the distances.
  Timer::Start("knn");
  typedef DTNNKMeansRules<MetricType, TreeType> RuleType;
  RuleType rules(centroids, dataset, assignments, distances, metric,
      prunedPoints, oldFromNewCentroids, visited);

  // Now construct the traverser ourselves.
  typename TreeType::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*tree, *centroidTree);
  Timer::Stop("knn");

  Timer::Start("tree_mod");
  DecoalesceTree(*tree);
  Timer::Stop("tree_mod");

  Log::Info << "This iteration: " << rules.BaseCases() << " base cases, " <<
      rules.Scores() << " scores.\n";
  distanceCalculations += rules.BaseCases() + rules.Scores();

  // From the assignments, calculate the new centroids and counts.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (visited[i])
    {
      newCentroids.col(assignments(0, i)) += dataset.col(i);
      ++counts(assignments(0, i));
      // Reset for next iteration.
      visited[i] = false;
    }
  }

  newCentroids += prunedCentroids;
  counts += prunedCounts;

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
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

//  lastIterationCentroids = oldCentroids;

  delete centroidTree;

  ++iteration;

  return std::sqrt(residual);
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::UpdateTree(
    TreeType& node,
    const arma::mat& centroids,
    const arma::mat& interclusterDistances,
    const std::vector<size_t>& newFromOldCentroids)
{
  // Update iteration.
//  node.Stat().Iteration() = iteration;

  if (node.Stat().Owner() == size_t(-1))
    node.Stat().Owner() = centroids.n_cols;

  // Do the tree update in a depth-first manner: leaves first.
  bool childrenPruned = true;
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    UpdateTree(node.Child(i), centroids, interclusterDistances,
        newFromOldCentroids);
    if (!node.Child(i).Stat().Pruned())
      childrenPruned = false; // Not all children are pruned.
  }

  const bool prunedLastIteration = node.Stat().Pruned();

  // Does the node have a single owner?
  // It would be nice if we could do this during the traversal.
  bool singleOwner = true;
  size_t owner = centroids.n_cols + 1;
  if (!node.Stat().Pruned() && childrenPruned)
  {
    // Determine the bounds for the points.
    double newMaxClusterDistance = 0.0;
    double newSecondClusterBound = DBL_MAX;
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      // Don't forget to map back from the new cluster index.
      size_t c;
      if (!prunedPoints[node.Point(i)])
        c = assignments(0, node.Point(i));
      else
        c = lastOwners[node.Point(i)];

      if (owner == centroids.n_cols + 1)
        owner = c;
      else if (owner != c)
      {
        singleOwner = false;
        break;
      }

      // Update maximum cluster distance and second cluster bound.
      if (!prunedPoints[node.Point(i)])
      {
        if (distances(0, node.Point(i)) > newMaxClusterDistance)
          newMaxClusterDistance = distances(0, node.Point(i));
        if (distances(1, node.Point(i)) < newSecondClusterBound)
          newSecondClusterBound = distances(1, node.Point(i));
      }
      else
      {
        // Use the cached bounds.
        if (distances(0, node.Point(i)) > newMaxClusterDistance)
          newMaxClusterDistance = distances(0, node.Point(i));
        if (lowerSecondBounds[node.Point(i)] < newSecondClusterBound)
          newSecondClusterBound = lowerSecondBounds[node.Point(i)];
      }
    }

    for (size_t i = 0; i < node.NumChildren(); ++i)
    {
      if (owner == centroids.n_cols + 1)
        owner = node.Child(i).Stat().Owner();
      else if ((node.Child(i).Stat().Owner() == centroids.n_cols) ||
               (owner != node.Child(i).Stat().Owner()))
      {
        singleOwner = false;
        break;
      }

      // Update maximum cluster distance and second cluster bound.
      if (node.Child(i).Stat().MaxClusterDistance() > newMaxClusterDistance)
        newMaxClusterDistance = node.Child(i).Stat().MaxClusterDistance();
      if (node.Child(i).Stat().SecondClusterBound() < newSecondClusterBound)
        newSecondClusterBound = node.Child(i).Stat().SecondClusterBound();
    }

    // Okay, now we know if it's owned or not, and by which cluster.
    if (singleOwner)
    {
      node.Stat().Owner() = owner;

      // What do we do with the new cluster bounds?
      if (newMaxClusterDistance > 0.0 && newMaxClusterDistance <
          node.Stat().MaxClusterDistance())
        node.Stat().MaxClusterDistance() = newMaxClusterDistance;
      if (newSecondClusterBound != DBL_MAX && newSecondClusterBound >
          node.Stat().SecondClusterBound())
        node.Stat().SecondClusterBound() = newSecondClusterBound;

      // Convenience variables to clean up the expressions.
      const double mcd = node.Stat().MaxClusterDistance();
      const double scb = node.Stat().SecondClusterBound();
      const double ownerMovement = clusterDistances[owner];
      const double maxMovement = clusterDistances[centroids.n_cols];
      const double closestClusterDistance =
          interclusterDistances[newFromOldCentroids[owner]];
      if ((node.NumPoints() == 0 && childrenPruned) ||
          (mcd + ownerMovement < scb - maxMovement) ||
          (mcd < 0.5 * closestClusterDistance))
        node.Stat().Pruned() = true;

      if (!node.Stat().Pruned() && (mcd - ownerMovement) < (scb - maxMovement))
      {
        // Calculate the next MCD by hand.
        const double newDist = node.MaxDistance(centroids.col(owner));
        ++distanceCalculations;
        node.Stat().MaxClusterDistance() = newDist;

        if ((newDist < scb - maxMovement) ||
            (newDist < 0.5 * closestClusterDistance))
          node.Stat().Pruned() = true;
        else
          node.Stat().SecondClusterBound() -= maxMovement;
      }
      else
      {
        // Adjust bounds for next iteration, regardless of whether or not the
        // node was pruned.  (Does this adjustment need to happen if there is no
        // prune?
        node.Stat().MaxClusterDistance() += ownerMovement;
        node.Stat().SecondClusterBound() -= maxMovement;
      }
    }
    else if (childrenPruned && node.NumChildren() > 0 && node.NumPoints() == 0)
    {
      // The node isn't owned by a single cluster.  But if it has no points and
      // its children are all pruned, we may prune it too.
      node.Stat().Pruned() = true;
      node.Stat().Owner() = centroids.n_cols;
    }
  }
  else if (node.Stat().Pruned())
  {
    // The node was pruned last iteration.  See if the node can remain pruned.
    singleOwner = false;

    // If it was pruned because all points were pruned, we need to check
    // individually.
    if (node.Stat().Owner() == centroids.n_cols)
    {
      node.Stat().Pruned() = false;
    }
    else
    { 
      // Will our bounds still work?
      if (node.Stat().MaxClusterDistance() +
          clusterDistances[node.Stat().Owner()] <
          node.Stat().SecondClusterBound() - clusterDistances[centroids.n_cols])
      {
        // The node remains pruned.  Adjust the bounds for next iteration.
        node.Stat().MaxClusterDistance() +=
            clusterDistances[node.Stat().Owner()];
        node.Stat().SecondClusterBound() -= clusterDistances[centroids.n_cols];
      }
      else
      {
        // Attempt other prune.
        if (node.Stat().MaxClusterDistance() < 0.5 *
            interclusterDistances[newFromOldCentroids[node.Stat().Owner()]])
        {
          // The node remains pruned.  Adjust the bounds for next iteration.
          node.Stat().MaxClusterDistance() +=
              clusterDistances[node.Stat().Owner()];
          node.Stat().SecondClusterBound() -= clusterDistances[centroids.n_cols];
        }
        else
        {
          node.Stat().Pruned() = false;
          node.Stat().MaxClusterDistance() = DBL_MAX;
          node.Stat().SecondClusterBound() = 0.0;
        }
      }
    }
  }
  else
  {
    // The children haven't been pruned, so we can't.
    // This node was not pruned last iteration, so we simply need to adjust the
    // bounds.
    node.Stat().Owner() = centroids.n_cols;
    if (node.Stat().MaxClusterDistance() != DBL_MAX)
      node.Stat().MaxClusterDistance() += clusterDistances[centroids.n_cols];
    if (node.Stat().SecondClusterBound() != DBL_MAX)
      node.Stat().SecondClusterBound() = std::max(0.0,
          node.Stat().SecondClusterBound() -
          clusterDistances[centroids.n_cols]);
  }

  // If the node wasn't pruned, try to prune individual points.
  if (!node.Stat().Pruned())
  {
    bool allPruned = true;
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      const size_t index = node.Point(i);
      size_t owner;
      if (prunedLastIteration && node.Stat().Owner() < centroids.n_cols)
        owner = node.Stat().Owner();
      else
        owner = assignments(0, index);

      // Update lower bound, if possible.
      if (!prunedLastIteration && !prunedPoints[index])
        lowerSecondBounds[index] = distances(1, index);

      const double upperPointBound = distances(0, index) +
          clusterDistances[owner];
      const double lowerSecondBound = lowerSecondBounds[index] -
          clusterDistances[centroids.n_cols];
      const double closestClusterDistance =
          interclusterDistances[newFromOldCentroids[owner]];
      if ((upperPointBound < lowerSecondBound) ||
          (upperPointBound < 0.5 * closestClusterDistance))
      {
        prunedPoints[index] = true;
        distances(0, index) += clusterDistances[owner];
        lastOwners[index] = owner;
        distances(1, index) += clusterDistances[centroids.n_cols];
        lowerSecondBounds[index] -= clusterDistances[centroids.n_cols];
        prunedCentroids.col(owner) += dataset.col(index);
        prunedCounts(owner)++;
      }
      else
      {
        // Attempt to tighten the lower bound.
        distances(0, index) = metric.Evaluate(centroids.col(owner),
                                             dataset.col(index));
        ++distanceCalculations;

        if ((distances(0, index) < lowerSecondBound) ||
            (distances(0, index) < 0.5 * closestClusterDistance))
        {
          prunedPoints[index] = true;
          lastOwners[index] = owner;
          lowerSecondBounds[index] -= clusterDistances[centroids.n_cols];
          distances(1, index) += clusterDistances[centroids.n_cols];
          prunedCentroids.col(owner) += dataset.col(index);
          prunedCounts(owner)++;
        }
        else
        {
          prunedPoints[index] = false;
          allPruned = false;
          // Still update these anyway.
          distances(1, index) += clusterDistances[centroids.n_cols];
        }
      }
    }

    if (allPruned && node.NumPoints() > 0)
    {
      // Prune the entire node.
      node.Stat().Pruned() = true;
      node.Stat().Owner() = centroids.n_cols;
    }
  }

  if (node.Stat().Pruned())
  {
    // Update bounds.
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      const size_t index = node.Point(i);
      lowerSecondBounds[index] -= clusterDistances[node.Stat().Owner()];
    }
  }

  // Make sure all the point bounds are updated.
  for (size_t i = 0; i < node.NumPoints(); ++i)
  {
    const size_t index = node.Point(i);
    distances(0, index) += clusterDistances[assignments(0, index)];
    distances(1, index) += clusterDistances[assignments(1, index)];
  }

  if (node.Stat().FirstBound() != DBL_MAX)
    node.Stat().FirstBound() += clusterDistances[centroids.n_cols];
  if (node.Stat().SecondBound() != DBL_MAX)
    node.Stat().SecondBound() += clusterDistances[centroids.n_cols];
  if (node.Stat().Bound() != DBL_MAX)
    node.Stat().Bound() += clusterDistances[centroids.n_cols];
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::CoalesceTree(
    TreeType& node,
    const size_t child /* Which child are we? */)
{
  // If one of the two children is pruned, we hide this node.
  // This assumes the BinarySpaceTree.  (bad Ryan! bad!)
  if (node.NumChildren() == 0)
    return; // We can't do anything.

  // If this is the root node, we can't coalesce.
  if (node.Parent() != NULL)
  {
    if (node.Child(0).Stat().Pruned() && !node.Child(1).Stat().Pruned())
    {
      CoalesceTree(node.Child(1), 1);

      // Link the right child to the parent.
      node.Child(1).Parent() = node.Parent();
      node.Parent()->ChildPtr(child) = node.ChildPtr(1);
    }
    else if (!node.Child(0).Stat().Pruned() && node.Child(1).Stat().Pruned())
    {
      CoalesceTree(node.Child(0), 0);

      // Link the left child to the parent.
      node.Child(0).Parent() = node.Parent();
      node.Parent()->ChildPtr(child) = node.ChildPtr(0);

    }
    else if (!node.Child(0).Stat().Pruned() && !node.Child(1).Stat().Pruned())
    {
      // The conditional is probably not necessary.
      CoalesceTree(node.Child(0), 0);
      CoalesceTree(node.Child(1), 1);
    }
  }
  else
  {
    CoalesceTree(node.Child(0), 0);
    CoalesceTree(node.Child(1), 1);
  }
}

template<typename MetricType, typename MatType, typename TreeType>
void DTNNKMeans<MetricType, MatType, TreeType>::DecoalesceTree(TreeType& node)
{
  node.Parent() = (TreeType*) node.Stat().TrueParent();
  node.ChildPtr(0) = (TreeType*) node.Stat().TrueLeft();
  node.ChildPtr(1) = (TreeType*) node.Stat().TrueRight();

  if (node.NumChildren() > 0)
  {
    DecoalesceTree(node.Child(0));
    DecoalesceTree(node.Child(1));
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
