/**
 * @file core/tree/rectangle_tree/r_star_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (RStarTreeSplit) to split a RectangleTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_IMPL_HPP

#include "r_star_tree_split.hpp"
#include "rectangle_tree.hpp"

#include <mlpack/core/math/range.hpp>

namespace mlpack {

/**
 * Reinsert any points into the tree, if needed.  This returns the number of
 * points reinserted.
 */
template<typename TreeType>
size_t RStarTreeSplit::ReinsertPoints(TreeType* tree,
                                      std::vector<bool>& relevels)
{
  // Convenience typedef.
  using ElemType = typename TreeType::ElemType;

  // Check if we need to reinsert.
  if (relevels[tree->TreeDepth() - 1])
  {
    relevels[tree->TreeDepth() - 1] = false;

    // We sort the points by decreasing distance to the centroid of the bound.
    // We then remove the first p entries and reinsert them at the root.
    TreeType* root = tree;
    while (root->Parent() != NULL)
      root = root->Parent();
    size_t p = tree->MaxLeafSize() * 0.3; // The paper says this works the best.
    if (p > 0)
    {
      // We'll only do reinsertions if p > 0.  p is the number of points we will
      // reinsert.  If p == 0, then no reinsertion is necessary and we continue
      // with the splitting procedure.
      std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
      arma::Col<ElemType> center;
      tree->Bound().Center(center);
      for (size_t i = 0; i < sorted.size(); ++i)
      {
        sorted[i].first = tree->Distance().Evaluate(center,
            tree->Dataset().col(tree->Point(i)));
        sorted[i].second = tree->Point(i);
      }

      std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);

      // Remove the points furthest from the center of the node.
      for (size_t i = 0; i < p; ++i)
        root->DeletePoint(sorted[sorted.size() - 1 - i].second, relevels);

      // Now reinsert the points, but reverse the order---insert the closest to
      // the center first.
      for (size_t i = p; i > 0; --i)
        root->InsertPoint(sorted[sorted.size() - i].second, relevels);
    }

    return p;
  }

  return 0;
}

/**
 * Given a node, return the best dimension and the best index to split on.
 */
template<typename TreeType>
void RStarTreeSplit::PickLeafSplit(TreeType* tree,
                                   size_t& bestAxis,
                                   size_t& bestIndex)
{
  // Convenience typedef.
  using ElemType = typename TreeType::ElemType;
  using BoundType = HRectBound<EuclideanDistance, ElemType>;

  bestAxis = 0;
  bestIndex = 0;
  ElemType bestScore = std::numeric_limits<ElemType>::max();

  /**
   * Check each dimension, to find which dimension is best to split on.
   */
  for (size_t j = 0; j < tree->Bound().Dim(); ++j)
  {
    ElemType axisScore = 0.0;

    // Sort in increasing values of the selected dimension j.
    arma::Col<ElemType> dimValues(tree->Count());
    for (size_t i = 0; i < tree->Count(); ++i)
      dimValues[i] = tree->Dataset().col(tree->Point(i))[j];
    arma::uvec sortedIndices = arma::sort_index(dimValues);

    // We'll store each of the three scores for each distribution.
    const size_t numPossibleSplits = tree->MaxLeafSize() -
        2 * tree->MinLeafSize() + 2;
    arma::Col<ElemType> areas(numPossibleSplits);
    arma::Col<ElemType> margins(numPossibleSplits);
    arma::Col<ElemType> overlaps(numPossibleSplits);

    for (size_t i = 0; i < numPossibleSplits; ++i)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinLeafSize() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.
      size_t splitIndex = tree->MinLeafSize() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < splitIndex; l++)
        bound1 |= tree->Dataset().col(tree->Point(sortedIndices[l]));

      for (size_t l = splitIndex; l < tree->Count(); l++)
        bound2 |= tree->Dataset().col(tree->Point(sortedIndices[l]));

      areas[i] = bound1.Volume() + bound2.Volume();
      overlaps[i] = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); ++k)
        margins[i] += bound1[k].Width() + bound2[k].Width();

      axisScore += margins[i];
    }

    // Is this dimension a new best score?  We want the lowest possible score.
    if (axisScore < bestScore)
    {
      bestScore = axisScore;
      bestAxis = j;
      size_t overlapIndex = 0;
      size_t areaIndex = 0;
      bool tiedOnOverlap = false;

      for (size_t i = 1; i < areas.n_elem; ++i)
      {
        if (overlaps[i] < overlaps[overlapIndex])
        {
          tiedOnOverlap = false;
          overlapIndex = i;
          areaIndex = i;
        }
        else if (overlaps[i] == overlaps[overlapIndex])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[areaIndex])
            areaIndex = i;
        }
      }

      // Select the best index for splitting.
      bestIndex = (tiedOnOverlap ? areaIndex : overlapIndex);
    }
  }
}

/**
 * We call GetPointSeeds to get the two points which will be the initial points
 * in the new nodes We then call AssignPointDestNode to assign the remaining
 * points to the two new nodes.  Finally, we delete the old node and insert the
 * new nodes into the tree, spliting the parent if necessary.
 */
template<typename TreeType>
void RStarTreeSplit::SplitLeafNode(TreeType *tree, std::vector<bool>& relevels)
{
  // Convenience typedef.
  using ElemType = typename TreeType::ElemType;

  // If there's no need to split, don't.
  if (tree->Count() <= tree->MaxLeafSize())
    return;

  // If we haven't yet checked if we need to reinsert on this level, we try
  // doing so now.
  if (ReinsertPoints(tree, relevels) > 0)
    return;

  // We don't need to reinsert.  Instead, we need to split the node.
  size_t bestAxis;
  size_t bestIndex;
  PickLeafSplit(tree, bestAxis, bestIndex);

  /**
   * Now that we have found the best dimension to split on, re-sort in that
   * dimension to prepare for reinsertion of points into the new nodes.
   */
  std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
  for (size_t i = 0; i < sorted.size(); ++i)
  {
    sorted[i].first = tree->Dataset().col(tree->Point(i))[bestAxis];
    sorted[i].second = tree->Point(i);
  }

  std::sort(sorted.begin(), sorted.end(), PairComp<ElemType, size_t>);

  /**
   * If 'tree' is the root of the tree (i.e. if it has no parent), then we must
   * create two new child nodes, distribute the points from the original node
   * among them, and insert those.  If 'tree' is not the root of the tree, then
   * we may create only one new child node, redistribute the points from the
   * original node between 'tree' and the new node, then insert those nodes into
   * the parent.
   *
   * Here we simply set treeOne and treeTwo to the right values to avoid code
   * duplication.
   */
  TreeType* par = tree->Parent();
  TreeType* treeOne = (par) ? tree              : new TreeType(tree);
  TreeType* treeTwo = (par) ? new TreeType(par) : new TreeType(tree);

  // Now clean the node, and we will re-use this.
  const size_t numPoints = tree->Count();

  // Reset the original node's values, regardless of whether it will become
  // the new parent or not.
  tree->numChildren = 0;
  tree->numDescendants = 0;
  tree->count = 0;
  tree->bound.Clear();

  // Insert the points into the appropriate tree.
  for (size_t i = 0; i < numPoints; ++i)
  {
    if (i < bestIndex + tree->MinLeafSize())
      treeOne->InsertPoint(sorted[i].second);
    else
      treeTwo->InsertPoint(sorted[i].second);
  }

  // Insert the new tree node(s).
  if (par)
  {
    // Just insert the new node into the parent.
    par->children[par->NumChildren()++] = treeTwo;

    // If we have overflowed the parent's children, then we need to split that
    // node also.
    if (par->NumChildren() == par->MaxNumChildren() + 1)
      RStarTreeSplit::SplitNonLeafNode(par, relevels);
  }
  else
  {
    // Now insert the two nodes into 'tree', which is now a higher-level root
    // node in the tree.
    InsertNodeIntoTree(tree, treeOne);
    InsertNodeIntoTree(tree, treeTwo);
  }
}

/**
 * We call GetBoundSeeds to get the two new nodes that this one will be broken
 * into.  Then we call AssignNodeDestNode to move the children of this node into
 * either of those two nodes.  Finally, we delete the now unused information and
 * recurse up the tree if necessary.  We don't need to worry about the bounds
 * higher up the tree because they were already updated if necessary.
 */
template<typename TreeType>
bool RStarTreeSplit::SplitNonLeafNode(
    TreeType *tree,
    std::vector<bool>& relevels)
{
  // Convenience typedef.
  using ElemType = typename TreeType::ElemType;
  using BoundType = HRectBound<EuclideanDistance, ElemType>;

  // Reinsertion isn't done for non-leaf nodes; the paper doesn't seem to make
  // it clear how to reinsert an entire node without reinserting each of the
  // points, so we will avoid that here.  This is a possible area for
  // improvement of this code.

  size_t bestAxis = 0;
  size_t bestIndex = 0;
  ElemType bestScore = std::numeric_limits<ElemType>::max();
  bool lowIsBetter = true;

  /**
   * Check over each dimension to see which is best to use for splitting.
   */
  for (size_t j = 0; j < tree->Bound().Dim(); ++j)
  {
    ElemType axisLoScore = 0.0;
    ElemType axisHiScore = 0.0;

    // We have to calculate values for both the lower and higher parts of the
    // bound.
    arma::Col<ElemType> loDimValues(tree->NumChildren());
    arma::Col<ElemType> hiDimValues(tree->NumChildren());
    for (size_t i = 0; i < tree->NumChildren(); ++i)
    {
      loDimValues[i] = tree->Child(i).Bound()[j].Lo();
      hiDimValues[i] = tree->Child(i).Bound()[j].Hi();
    }
    arma::uvec sortedLoDimIndices = arma::sort_index(loDimValues);
    arma::uvec sortedHiDimIndices = arma::sort_index(hiDimValues);

    // We'll store each of the three scores for each distribution.  Remember
    // that these are the sums calculated over both the low and high bounds of
    // each rectangle.
    const size_t numPossibleSplits = tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2;
    arma::Col<ElemType> areas(2 * numPossibleSplits);
    arma::Col<ElemType> margins(2 * numPossibleSplits);
    arma::Col<ElemType> overlaps(2 * numPossibleSplits);

    for (size_t i = 0; i < numPossibleSplits; ++i)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinNumChildren() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.
      const size_t splitIndex = tree->MinNumChildren() + i;

      BoundType lb1(tree->Bound().Dim());
      BoundType lb2(tree->Bound().Dim());
      BoundType hb1(tree->Bound().Dim());
      BoundType hb2(tree->Bound().Dim());

      for (size_t l = 0; l < splitIndex; ++l)
      {
        lb1 |= tree->Child(sortedLoDimIndices[l]).Bound();
        hb1 |= tree->Child(sortedHiDimIndices[l]).Bound();
      }
      for (size_t l = splitIndex; l < tree->NumChildren(); ++l)
      {
        lb2 |= tree->Child(sortedLoDimIndices[l]).Bound();
        hb2 |= tree->Child(sortedHiDimIndices[l]).Bound();
      }

      // Calculate low bound distributions.
      areas[2 * i] = lb1.Volume() + lb2.Volume();
      overlaps[2 * i] = lb1.Overlap(lb2);

      // Calculate high bound distributions.
      areas[2 * i + 1] = hb1.Volume() + hb2.Volume();
      overlaps[2 * i + 1] = hb1.Overlap(hb2);

      // Now calculate margins for each.
      for (size_t k = 0; k < lb1.Dim(); ++k)
      {
        margins[2 * i]     += lb1[k].Width() + lb2[k].Width();
        margins[2 * i + 1] += hb1[k].Width() + hb2[k].Width();
      }

      // The score we use is the sum of all scores.
      axisLoScore += margins[2 * i];
      axisHiScore += margins[2 * i + 1];
    }

    // If this dimension's score (for lower or higher bound scores) is a new
    // best, then extract the necessary split information.
    if (std::min(axisLoScore, axisHiScore) < bestScore)
    {
      bestScore = std::min(axisLoScore, axisHiScore);
      if (axisLoScore < axisHiScore)
        lowIsBetter = true;
      else
        lowIsBetter = false;

      bestAxis = j;

      // This will get us either the lower or higher bound depending on which we
      // want.  Remember that we selected *either* the lower or higher bounds to
      // split on, so we want to only check those.
      const size_t indexOffset = lowIsBetter ? 0 : 1;

      size_t overlapIndex = indexOffset;
      size_t areaIndex = indexOffset;
      bool tiedOnOverlap = false;

      // Find the best possible split (and whether it is on the low values or
      // high values of the bounds).
      for (size_t i = 1; i < numPossibleSplits; ++i)
      {
        // Check bounds.
        if (overlaps[2 * i + indexOffset] < overlaps[overlapIndex])
        {
          tiedOnOverlap = false;
          areaIndex = 2 * i + indexOffset;
          overlapIndex = 2 * i + indexOffset;
        }
        else if (overlaps[i] == overlaps[overlapIndex])
        {
          tiedOnOverlap = true;
          if (areas[2 * i + indexOffset] < areas[areaIndex])
            areaIndex = 2 * i + indexOffset;
        }
      }

      bestIndex = ((tiedOnOverlap ? areaIndex : overlapIndex) - indexOffset) / 2
          + tree->MinNumChildren();
    }
  }

  // Get a list of the old children.
  std::vector<TreeType*> oldChildren(tree->NumChildren());
  for (size_t i = 0; i < oldChildren.size(); ++i)
    oldChildren[i] = &tree->Child(i);

  /**
   * If 'tree' is the root of the tree (i.e. if it has no parent), then we must
   * create two new child nodes, distribute the children from the original node
   * among them, and insert those.  If 'tree' is not the root of the tree, then
   * we may create only one new child node, redistribute the children from the
   * original node between 'tree' and the new node, then insert those nodes into
   * the parent.
   *
   * Here, we simply set treeOne and treeTwo to the right values to avoid code
   * duplication.
   */
  TreeType* par = tree->Parent();
  TreeType* treeOne = par ? tree              : new TreeType(tree);
  TreeType* treeTwo = par ? new TreeType(par) : new TreeType(tree);

  // Now clean the node.
  tree->numChildren = 0;
  tree->numDescendants = 0;
  tree->count = 0;
  tree->bound.Clear();

  // Assemble vector of values.
  arma::Col<ElemType> values(oldChildren.size());
  for (size_t i = 0; i < oldChildren.size(); ++i)
  {
    values[i] = (lowIsBetter ? oldChildren[i]->Bound()[bestAxis].Lo()
                             : oldChildren[i]->Bound()[bestAxis].Hi());
  }
  arma::uvec indices = arma::sort_index(values);

  for (size_t i = 0; i < bestIndex; ++i)
    InsertNodeIntoTree(treeOne, oldChildren[indices[i]]);
  for (size_t i = bestIndex; i < oldChildren.size(); ++i)
    InsertNodeIntoTree(treeTwo, oldChildren[indices[i]]);

  // Insert the new tree node(s).
  if (par)
  {
    // Insert the new node into the parent.  The number of descendants does not
    // need to be updated.
    par->children[par->NumChildren()++] = treeTwo;
  }
  else
  {
    // Insert both nodes into 'tree', which is now a higher-level root node.
    InsertNodeIntoTree(tree, treeOne);
    InsertNodeIntoTree(tree, treeTwo);

    // We have to update the children of treeOne so that they record the correct
    // parent.
    for (size_t i = 0; i < treeOne->NumChildren(); ++i)
      treeOne->children[i]->Parent() = treeOne;
  }

  // Update the children of treeTwo to have the correct parent.
  for (size_t i = 0; i < treeTwo->NumChildren(); ++i)
    treeTwo->children[i]->Parent() = treeTwo;

  // If we have overflowed hte parent's children, then we need to split that
  // node also.
  if (par && par->NumChildren() >= par->MaxNumChildren() + 1)
    RStarTreeSplit::SplitNonLeafNode(par, relevels);

  return false;
}

/**
 * Insert a node into another node.  Expanding the bounds and updating the
 * numberOfChildren.
 */
template<typename TreeType>
void RStarTreeSplit::InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->numDescendants += srcNode->numDescendants;
  destTree->children[destTree->NumChildren()++] = srcNode;
}

} // namespace mlpack

#endif
