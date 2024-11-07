/**
 * @file methods/kmeans/dual_tree_kmeans_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of a Lloyd iteration which uses dual-tree nearest neighbor
 * search as a black box.  The conditions under which this will perform best are
 * probably limited to the case where k is close to the number of points in the
 * dataset, and the number of iterations of the k-means algorithm will be few.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_DTNN_KMEANS_IMPL_HPP
#define MLPACK_METHODS_KMEANS_DTNN_KMEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "dual_tree_kmeans.hpp"

#include "dual_tree_kmeans_rules.hpp"

namespace mlpack {

//! Call the tree constructor that does mapping.
template<typename TreeType, typename MatType>
TreeType* BuildForcedLeafSizeTree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    const std::enable_if_t<TreeTraits<TreeType>::RearrangesDataset>* = 0)
{
  // This is a hack.  I know this will be BinarySpaceTree, so force a leaf size
  // of one.
  return new TreeType(std::forward<MatType>(dataset), oldFromNew, 1);
}

//! Call the tree constructor that does not do mapping.
template<typename TreeType, typename MatType>
TreeType* BuildForcedLeafSizeTree(
    MatType&& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const std::enable_if_t<!TreeTraits<TreeType>::RearrangesDataset>* = 0)
{
  return new TreeType(std::forward<MatType>(dataset));
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
DualTreeKMeans<DistanceType, MatType, TreeType>::DualTreeKMeans(
    const MatType& dataset,
    DistanceType& distance) :
    datasetOrig(dataset),
    tree(new Tree(const_cast<MatType&>(dataset))),
    dataset(tree->Dataset()),
    distance(distance),
    distanceCalculations(0),
    iteration(0),
    upperBounds(dataset.n_cols),
    lowerBounds(dataset.n_cols),
    prunedPoints(dataset.n_cols, false), // Fill with false.
    assignments(dataset.n_cols),
    visited(dataset.n_cols, false) // Fill with false.
{
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    prunedPoints[i] = false;
    visited[i] = false;
  }
  assignments.fill(size_t(-1));
  upperBounds.fill(DBL_MAX);
  lowerBounds.fill(DBL_MAX);
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
DualTreeKMeans<DistanceType, MatType, TreeType>::~DualTreeKMeans()
{
  if (tree)
    delete tree;
}

// Run a single iteration.
template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
double DualTreeKMeans<DistanceType, MatType, TreeType>::Iterate(
    const arma::mat& centroids,
    arma::mat& newCentroids,
    arma::Col<size_t>& counts)
{
  // Build a tree on the centroids.  This will make a copy if necessary, which
  // is unfortunate, but I don't see a reasonable way around it.
  std::vector<size_t> oldFromNewCentroids;
  Tree* centroidTree = BuildForcedLeafSizeTree<Tree>(centroids,
      oldFromNewCentroids);

  // Find the nearest neighbors of each of the clusters.  We have to make our
  // own TreeType, which is a little bit abuse, but we know for sure the
  // TreeStatType we have will work.
  NeighborSearch<NearestNeighborSort, DistanceType, MatType, NNSTreeType>
      nns(std::move(*centroidTree));

  // Reset information in the tree, if we need to.
  if (iteration > 0)
  {
    // If the tree maps points, we need an intermediate result matrix.
    arma::mat* interclusterDistancesTemp = TreeTraits<Tree>::RearrangesDataset ?
        new arma::mat(1, centroids.n_elem) : &interclusterDistances;

    arma::Mat<size_t> closestClusters; // We don't actually care about these.
    nns.Search(1, closestClusters, *interclusterDistancesTemp);
    distanceCalculations += nns.BaseCases() + nns.Scores();

    // We need to do the unmapping ourselves, if the tree does mapping.
    if (TreeTraits<Tree>::RearrangesDataset)
    {
      for (size_t i = 0; i < interclusterDistances.n_elem; ++i)
        interclusterDistances[oldFromNewCentroids[i]] =
            (*interclusterDistancesTemp)[i];

      delete interclusterDistancesTemp;
    }

    UpdateTree(*tree, centroids);

    for (size_t i = 0; i < dataset.n_cols; ++i)
      visited[i] = false;
  }
  else
  {
    // Not initialized yet.
    clusterDistances.set_size(centroids.n_cols + 1);
    interclusterDistances.set_size(1, centroids.n_cols);
  }

  // We won't use the KNN class here because we have our own set of rules.
  lastIterationCentroids = centroids;
  using RuleType = DualTreeKMeansRules<DistanceType, Tree>;
  RuleType rules(nns.ReferenceTree().Dataset(), dataset, assignments,
      upperBounds, lowerBounds, distance, prunedPoints, oldFromNewCentroids,
      visited);

  typename Tree::template BreadthFirstDualTreeTraverser<RuleType>
      traverser(rules);

  CoalesceTree(*tree);

  // Set the number of pruned centroids in the root to 0.
  tree->Stat().Pruned() = 0;
  traverser.Traverse(*tree, nns.ReferenceTree());
  distanceCalculations += rules.BaseCases() + rules.Scores();

  DecoalesceTree(*tree);

  // Now we need to extract the clusters.
  newCentroids.zeros(centroids.n_rows, centroids.n_cols);
  counts.zeros(centroids.n_cols);
  ExtractCentroids(*tree, newCentroids, counts, centroids);

  // Now, calculate how far the clusters moved, after normalizing them.
  double residual = 0.0;
  clusterDistances[centroids.n_cols] = 0.0;
  for (size_t c = 0; c < centroids.n_cols; ++c)
  {
    if (counts[c] == 0)
    {
      clusterDistances[c] = 0;
    }
    else
    {
      newCentroids.col(c) /= counts(c);
      const double movement = distance.Evaluate(centroids.col(c),
          newCentroids.col(c));
      clusterDistances[c] = movement;
      residual += std::pow(movement, 2.0);

      if (movement > clusterDistances[centroids.n_cols])
        clusterDistances[centroids.n_cols] = movement;
    }
  }
  distanceCalculations += centroids.n_cols;

  delete centroidTree;

  ++iteration;

  return std::sqrt(residual);
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void DualTreeKMeans<DistanceType, MatType, TreeType>::UpdateTree(
    Tree& node,
    const arma::mat& centroids,
    const double parentUpperBound,
    const double adjustedParentUpperBound,
    const double parentLowerBound,
    const double adjustedParentLowerBound)
{
  const bool prunedLastIteration = node.Stat().StaticPruned();
  node.Stat().StaticPruned() = false;

  // Grab information from the parent, if we can.
  if (node.Parent() != NULL &&
      node.Parent()->Stat().Pruned() == centroids.n_cols &&
      node.Parent()->Stat().Owner() < centroids.n_cols)
  {
    // When taking bounds from the parent, note that the parent has already
    // adjusted the bounds according to the cluster movements, so we need to
    // de-adjust them since we'll adjust them again.  Maybe there is a smarter
    // way to do this...
    node.Stat().UpperBound() = parentUpperBound;
    node.Stat().LowerBound() = parentLowerBound;
    node.Stat().Pruned() = node.Parent()->Stat().Pruned();
    node.Stat().Owner() = node.Parent()->Stat().Owner();
  }
  const double unadjustedUpperBound = node.Stat().UpperBound();
  double adjustedUpperBound = adjustedParentUpperBound;
  const double unadjustedLowerBound = node.Stat().LowerBound();
  double adjustedLowerBound = adjustedParentLowerBound;

  if ((node.Stat().Pruned() == centroids.n_cols) &&
      (node.Stat().Owner() < centroids.n_cols))
  {
    // Adjust bounds.
    node.Stat().UpperBound() += clusterDistances[node.Stat().Owner()];
    node.Stat().LowerBound() -= clusterDistances[centroids.n_cols];

    if (adjustedParentUpperBound < node.Stat().UpperBound())
      node.Stat().UpperBound() = adjustedParentUpperBound;

    if (adjustedParentLowerBound > node.Stat().LowerBound())
      node.Stat().LowerBound() = adjustedParentLowerBound;

    // Try to use the inter-cluster distances to produce a better lower bound,
    // if possible.
    const double interclusterBound = interclusterDistances[node.Stat().Owner()]
        / 2.0;
    if (interclusterBound > node.Stat().LowerBound())
    {
      node.Stat().LowerBound() = interclusterBound;
      adjustedLowerBound = node.Stat().LowerBound();
    }

    if (node.Stat().UpperBound() < node.Stat().LowerBound())
    {
      node.Stat().StaticPruned() = true;
    }
    else
    {
      // Tighten bound.
      node.Stat().UpperBound() =
          std::min(node.Stat().UpperBound(),
                   node.MaxDistance(centroids.col(node.Stat().Owner())));
      adjustedUpperBound = node.Stat().UpperBound();

      ++distanceCalculations;
      if (node.Stat().UpperBound() < node.Stat().LowerBound())
        node.Stat().StaticPruned() = true;
    }
  }
  else
  {
    node.Stat().LowerBound() -= clusterDistances[centroids.n_cols];
  }

  // Recurse into children, and if all the children (and all the points) are
  // pruned, then we can mark this as statically pruned.
  bool allChildrenPruned = true;
  for (size_t i = 0; i < node.NumChildren(); ++i)
  {
    UpdateTree(node.Child(i), centroids, unadjustedUpperBound,
        adjustedUpperBound, unadjustedLowerBound, adjustedLowerBound);
    if (!node.Child(i).Stat().StaticPruned())
      allChildrenPruned = false;
  }

  bool allPointsPruned = true;
  if (TreeTraits<Tree>::HasSelfChildren && node.NumChildren() > 0)
  {
    // If this tree type has self-children, then we have already adjusted the
    // point bounds at a lower level, and we can determine if all of our points
    // are pruned simply by seeing if all of the children's points are pruned.
    // This particular line below additionally assumes that each node's points
    // are all contained in its first child.  This is valid for the cover tree,
    // but maybe not others.
    allPointsPruned = node.Child(0).Stat().StaticPruned();
  }
  else if (!node.Stat().StaticPruned())
  {
    // Try to prune individual points.
    for (size_t i = 0; i < node.NumPoints(); ++i)
    {
      const size_t index = node.Point(i);
      if (!visited[index] && !prunedPoints[index])
      {
        upperBounds[index] = DBL_MAX; // Reset the bounds.
        lowerBounds[index] = DBL_MAX;
        allPointsPruned = false;
        continue; // We didn't visit it and we don't have valid bounds -- so we
                  // can't prune it.
      }

      if (prunedLastIteration)
      {
        // It was pruned last iteration but not this iteration.
        // Set the bounds correctly.
        upperBounds[index] += node.Stat().StaticUpperBoundMovement();
        lowerBounds[index] -= node.Stat().StaticLowerBoundMovement();
      }

      prunedPoints[index] = false;
      const size_t owner = assignments[index];
      const double lowerBound = std::min(lowerBounds[index] -
          clusterDistances[centroids.n_cols], node.Stat().LowerBound());
      const double pruningLowerBound = std::max(lowerBound,
          interclusterDistances[owner] / 2.0);
      if (upperBounds[index] + clusterDistances[owner] < pruningLowerBound)
      {
        prunedPoints[index] = true;
        upperBounds[index] += clusterDistances[owner];
        lowerBounds[index] = pruningLowerBound;
      }
      else
      {
        // Attempt to tighten the bound.
        upperBounds[index] = distance.Evaluate(dataset.col(index),
                                               centroids.col(owner));
        ++distanceCalculations;
        if (upperBounds[index] < pruningLowerBound)
        {
          prunedPoints[index] = true;
          lowerBounds[index] = pruningLowerBound;
        }
        else
        {
          // Point cannot be pruned.  We may have to inspect the point at a
          // lower level, though.  If that's the case, then we shouldn't
          // invalidate the bounds we've got -- it will happen at the lower
          // level.
          if (!TreeTraits<Tree>::HasSelfChildren ||
              node.NumChildren() == 0)
          {
            upperBounds[index] = DBL_MAX;
            lowerBounds[index] = DBL_MAX;
          }
          allPointsPruned = false;
        }
      }
    }
  }

/*
  if (node.Stat().StaticPruned() && !allChildrenPruned)
  {
    Log::Warn << node;
    for (size_t i = 0; i < node.NumChildren(); ++i)
      Log::Warn << "child " << i << ":\n" << node.Child(i);
    Log::Fatal << "Node is statically pruned but not all its children are!\n";
  }
*/

  // If all of the children and points are pruned, we may mark this node as
  // pruned.
  if (allChildrenPruned && allPointsPruned && !node.Stat().StaticPruned())
  {
    node.Stat().StaticPruned() = true;
    node.Stat().Owner() = centroids.n_cols; // Invalid owner.
    node.Stat().Pruned() = size_t(-1);
  }

  if (!node.Stat().StaticPruned())
  {
    node.Stat().UpperBound() = DBL_MAX;
    node.Stat().LowerBound() = DBL_MAX;
    node.Stat().Pruned() = size_t(-1);
    node.Stat().Owner() = centroids.n_cols;
    node.Stat().StaticPruned() = false;
  }
  else // The node is now pruned.
  {
    if (prunedLastIteration)
    {
      // Track total movement while pruned.
      node.Stat().StaticUpperBoundMovement() +=
          clusterDistances[node.Stat().Owner()];
      node.Stat().StaticLowerBoundMovement() +=
          clusterDistances[centroids.n_cols];
    }
    else
    {
      node.Stat().StaticUpperBoundMovement() =
          clusterDistances[node.Stat().Owner()];
      node.Stat().StaticLowerBoundMovement() =
          clusterDistances[centroids.n_cols];
    }
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void DualTreeKMeans<DistanceType, MatType, TreeType>::ExtractCentroids(
    Tree& node,
    arma::mat& newCentroids,
    arma::Col<size_t>& newCounts,
    const arma::mat& centroids)
{
  // Does this node own points?
  if ((node.Stat().Pruned() == newCentroids.n_cols) ||
      (node.Stat().StaticPruned() && node.Stat().Owner() < newCentroids.n_cols))
  {
    const size_t owner = node.Stat().Owner();
    newCentroids.col(owner) += node.Stat().Centroid() * node.NumDescendants();
    newCounts[owner] += node.NumDescendants();
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
      ExtractCentroids(node.Child(i), newCentroids, newCounts, centroids);
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void DualTreeKMeans<DistanceType, MatType, TreeType>::CoalesceTree(
    Tree& node,
    const size_t child /* Which child are we? */)
{
  // If all children except one are pruned, we can hide this node.
  if (node.NumChildren() == 0)
    return; // We can't do anything.

  // If this is the root node, we can't coalesce.
  if (node.Parent() != NULL)
  {
    // First, we should coalesce those nodes that aren't statically pruned.
    for (size_t i = node.NumChildren() - 1; i > 0; --i)
    {
      if (node.Child(i).Stat().StaticPruned())
        HideChild(node, i);
      else
        CoalesceTree(node.Child(i), i);
    }

    if (node.Child(0).Stat().StaticPruned())
      HideChild(node, 0);
    else
      CoalesceTree(node.Child(0), 0);

    // If we've pruned all but one child, then notPrunedIndex will contain the
    // index of that child, and we can coalesce this node entirely.  Note that
    // the case where all children are statically pruned should not happen,
    // because then this node should itself be statically pruned.
    if (node.NumChildren() == 1)
    {
      node.Child(0).Parent() = node.Parent();
      node.Parent()->ChildPtr(child) = node.ChildPtr(0);
    }
  }
  else
  {
    // We can't coalesce the root, so call the children individually and
    // coalesce them.
    for (size_t i = 0; i < node.NumChildren(); ++i)
      CoalesceTree(node.Child(i), i);
  }
}

template<typename DistanceType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void DualTreeKMeans<DistanceType, MatType, TreeType>::DecoalesceTree(Tree& node)
{
  node.Parent() = (Tree*) node.Stat().TrueParent();
  RestoreChildren(node);

  for (size_t i = 0; i < node.NumChildren(); ++i)
    DecoalesceTree(node.Child(i));
}

//! Utility function for hiding children in a non-binary tree.
template<typename TreeType>
void HideChild(TreeType& node,
               const size_t child,
               const typename std::enable_if_t<
                   !TreeTraits<TreeType>::BinaryTree>*)
{
  // We're going to assume we have a Children() function open to us.  If we
  // don't, then this won't work, I guess...
  node.Children().erase(node.Children().begin() + child);
}

//! Utility function for hiding children in a binary tree.
template<typename TreeType>
void HideChild(TreeType& node,
               const size_t child,
               const typename std::enable_if_t<
                   TreeTraits<TreeType>::BinaryTree>*)
{
  // If we're hiding the left child, then take the right child as the new left
  // child.
  if (child == 0)
  {
    node.ChildPtr(0) = node.ChildPtr(1);
    node.ChildPtr(1) = NULL;
  }
  else
  {
    node.ChildPtr(1) = NULL;
  }
}

//! Utility function for restoring children in a non-binary tree.
template<typename TreeType>
void RestoreChildren(TreeType& node,
                     const typename std::enable_if_t<
                         !TreeTraits<TreeType>::BinaryTree>*)
{
  node.Children().clear();
  node.Children().resize(node.Stat().NumTrueChildren());
  for (size_t i = 0; i < node.Stat().NumTrueChildren(); ++i)
    node.Children()[i] = (TreeType*) node.Stat().TrueChild(i);
}

//! Utility function for restoring children in a binary tree.
template<typename TreeType>
void RestoreChildren(TreeType& node,
                     const typename std::enable_if_t<
                         TreeTraits<TreeType>::BinaryTree>*)
{
  if (node.Stat().NumTrueChildren() > 0)
  {
    node.ChildPtr(0) = (TreeType*) node.Stat().TrueChild(0);
    node.ChildPtr(1) = (TreeType*) node.Stat().TrueChild(1);
  }
}

} // namespace mlpack

#endif
