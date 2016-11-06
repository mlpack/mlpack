/**
 * @file x_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (XTreeSplit) to split a RectangleTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_IMPL_HPP

#include "x_tree_split.hpp"
#include "rectangle_tree.hpp"
#include <mlpack/core/math/range.hpp>

namespace mlpack {
namespace tree {

/**
 * We call GetPointSeeds to get the two points which will be the initial points
 * in the new nodes We then call AssignPointDestNode to assign the remaining
 * points to the two new nodes.  Finally, we delete the old node and insert the
 * new nodes into the tree, spliting the parent if necessary.
 */
template<typename TreeType>
void XTreeSplit::SplitLeafNode(TreeType *tree,std::vector<bool>& relevels)
{
  // Convenience typedef.
  typedef typename TreeType::ElemType ElemType;
  typedef bound::HRectBound<metric::EuclideanDistance, ElemType> BoundType;

  if (tree->Count() <= tree->MaxLeafSize())
    return;

  // If we are splitting the root node, we need will do things differently so
  // that the constructor and other methods don't confuse the end user by giving
  // an address of another node.
  if (tree->Parent() == NULL)
  {
    // We actually want to copy this way.  Pointers and everything.
    TreeType* copy = new TreeType(*tree, false);
    copy->Parent() = tree;
    tree->Count() = 0;
    tree->NullifyData();
    // Because this was a leaf node, numChildren must be 0.
    tree->children[(tree->NumChildren())++] = copy;
    assert(tree->NumChildren() == 1);
    XTreeSplit::SplitLeafNode(copy,relevels);
    return;
  }

  // If we haven't yet reinserted on this level, we try doing so now.
  if (relevels[tree->TreeDepth()])
  {
    relevels[tree->TreeDepth()] = false;
    // We sort the points by decreasing distance to the centroid of the bound.
    // We then remove the first p entries and reinsert them at the root.
    TreeType* root = tree;
    while (root->Parent() != NULL)
      root = root->Parent();

    // The R*-tree paper says this works the best.
    size_t p = tree->MaxLeafSize() * 0.3;
    if (p == 0)
    {
      XTreeSplit::SplitLeafNode(tree,relevels);
      return;
    }

    std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
    arma::Col<ElemType> center;
    tree->Bound().Center(center); // Modifies centroid.
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Metric().Evaluate(center,
          tree->Dataset().col(tree->Point(i)));
       sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType>);
    std::vector<size_t> pointIndices(p);

    for (size_t i = 0; i < p; i++)
    {
      // We start from the end of sorted.
      pointIndices[i] = tree->Point(sorted[sorted.size() - 1 - i].second);

      root->DeletePoint(tree->Point(sorted[sorted.size() - 1 - i].second),
          relevels);
    }

    for (size_t i = 0; i < p; i++)
    {
      // We reverse the order again to reinsert the closest points first.
      root->InsertPoint(pointIndices[p - 1 - i], relevels);
    }

//    // If we went below min fill, delete this node and reinsert all points.
//    if (tree->Count() < tree->MinLeafSize()) {
//      std::vector<int> pointIndices(tree->Count());
//      for(size_t i = 0; i < tree->Count(); i++) {
//        pointIndices[i] = tree->Points()[i];
//      }
//      root->RemoveNode(tree, relevels);
//      for(size_t i = 0; i < pointIndices.size(); i++) {
//        root->InsertPoint(pointIndices[i], relevels);
//      }
//      //tree->SoftDelete();
//    }
    return;
  }

  int bestOverlapIndexOnBestAxis = 0;
  int bestAreaIndexOnBestAxis = 0;
  bool tiedOnOverlap = false;
  int bestAxis = 0;
  ElemType bestAxisScore = std::numeric_limits<ElemType>::max();
  for (size_t j = 0; j < tree->Bound().Dim(); j++)
  {
    ElemType axisScore = 0.0;
    // Since we only have points in the leaf nodes, we only need to sort once.
    std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Dataset().col(tree->Point(i))[j];
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType>);

    // We'll store each of the three scores for each distribution.
    std::vector<ElemType> areas(tree->MaxLeafSize() -
        2 * tree->MinLeafSize() + 2);
    std::vector<ElemType> margins(tree->MaxLeafSize() -
        2 * tree->MinLeafSize() + 2);
    std::vector<ElemType> overlapedAreas(tree->MaxLeafSize() -
        2 * tree->MinLeafSize() + 2);
    for (size_t i = 0; i < areas.size(); i++)
    {
      areas[i] = 0.0;
      margins[i] = 0.0;
      overlapedAreas[i] = 0.0;
    }
    for (size_t i = 0; i < areas.size(); i++)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinLeafSize() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.

      size_t cutOff = tree->MinLeafSize() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < cutOff; l++)
        bound1 |= tree->Dataset().col(tree->Point(sorted[l].second));

      for (size_t l = cutOff; l < tree->Count(); l++)
        bound2 |= tree->Dataset().col(tree->Point(sorted[l].second));

      ElemType area1 = bound1.Volume();
      ElemType area2 = bound2.Volume();
      ElemType oArea = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); k++)
        margins[i] += bound1[k].Width() + bound2[k].Width();

      areas[i] += area1 + area2;
      overlapedAreas[i] += oArea;
      axisScore += margins[i];
    }

    if (axisScore < bestAxisScore)
    {
      bestAxisScore = axisScore;
      bestAxis = j;
      bestOverlapIndexOnBestAxis = 0;
      bestAreaIndexOnBestAxis = 0;
      for (size_t i = 1; i < areas.size(); i++)
      {
        if (overlapedAreas[i] < overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = false;
          bestAreaIndexOnBestAxis = i;
          bestOverlapIndexOnBestAxis = i;
        }
        else if (overlapedAreas[i] ==
                 overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[bestAreaIndexOnBestAxis])
            bestAreaIndexOnBestAxis = i;
        }
      }
    }
  }

  std::vector<std::pair<ElemType, size_t>> sorted(tree->Count());
  for (size_t i = 0; i < sorted.size(); i++)
  {
    sorted[i].first = tree->Dataset().col(tree->Point(i))[bestAxis];
    sorted[i].second = i;
  }

  std::sort(sorted.begin(), sorted.end(), PairComp<ElemType>);

  TreeType* treeOne = new TreeType(tree->Parent(),
                            tree->AuxiliaryInfo().NormalNodeMaxNumChildren());
  TreeType* treeTwo = new TreeType(tree->Parent(),
                            tree->AuxiliaryInfo().NormalNodeMaxNumChildren());

  // The leaf nodes should never have any overlap introduced by the above method
  // since a split axis is chosen and then points are assigned based on their
  // value along that axis.
  if (tiedOnOverlap)
  {
    for (size_t i = 0; i < tree->Count(); i++)
    {
      if (i < bestAreaIndexOnBestAxis + tree->MinLeafSize())
        treeOne->InsertPoint(tree->Point(sorted[i].second));
      else
        treeTwo->InsertPoint(tree->Point(sorted[i].second));
    }
  }
  else
  {
    for (size_t i = 0; i < tree->Count(); i++)
    {
      if (i < bestOverlapIndexOnBestAxis + tree->MinLeafSize())
        treeOne->InsertPoint(tree->Point(sorted[i].second));
      else
        treeTwo->InsertPoint(tree->Point(sorted[i].second));
    }
  }

  // Remove this node and insert treeOne and treeTwo.
  TreeType* par = tree->Parent();
  size_t index = par->NumChildren();
  for (size_t i = 0; i < par->NumChildren(); i++)
  {
    if (par->children[i] == tree)
    {
      index = i;
      break;
    }
  }
  assert(index != par->NumChildren());
  par->children[index] = treeOne;
  par->children[par->NumChildren()++] = treeTwo;

  // We now update the split history of each new node.
  treeOne->AuxiliaryInfo().SplitHistory().history[bestAxis] = true;
  treeOne->AuxiliaryInfo().SplitHistory().lastDimension = bestAxis;
  treeTwo->AuxiliaryInfo().SplitHistory().history[bestAxis] = true;
  treeTwo->AuxiliaryInfo().SplitHistory().lastDimension = bestAxis;

  // We only add one at a time, so we should only need to test for equality just
  // in case, we use an assert.
  assert(par->NumChildren() <= par->MaxNumChildren() + 1);
  if (par->NumChildren() == par->MaxNumChildren() + 1)
    XTreeSplit::SplitNonLeafNode(par,relevels);

  assert(treeOne->Parent()->NumChildren() <=
      treeOne->Parent()->MaxNumChildren());
  assert(treeOne->Parent()->NumChildren() >=
      treeOne->Parent()->MinNumChildren());
  assert(treeTwo->Parent()->NumChildren() <=
      treeTwo->Parent()->MaxNumChildren());
  assert(treeTwo->Parent()->NumChildren() >=
      treeTwo->Parent()->MinNumChildren());

  tree->SoftDelete();
}

/**
 * We call GetBoundSeeds to get the two new nodes that this one will be broken
 * into.  Then we call AssignNodeDestNode to move the children of this node into
 * either of those two nodes.  Finally, we delete the now unused information and
 * recurse up the tree if necessary.  We don't need to worry about the bounds
 * higher up the tree because they were already updated if necessary.
 */
template<typename TreeType>
bool XTreeSplit::SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels)
{
  // Convenience typedef.
  typedef typename TreeType::ElemType ElemType;
  typedef bound::HRectBound<metric::EuclideanDistance, ElemType> BoundType;

  // If we are splitting the root node, we need will do things differently so
  // that the constructor and other methods don't confuse the end user by giving
  // an address of another node.
  if (tree->Parent() == NULL)
  {
    // We actually want to copy this way.  Pointers and everything.
    TreeType* copy = new TreeType(*tree, false);

    copy->Parent() = tree;
    tree->NumChildren() = 0;
    tree->NullifyData();
    tree->children[(tree->NumChildren())++] = copy;
    XTreeSplit::SplitNonLeafNode(copy,relevels);
    return true;
  }

  // The X tree paper doesn't explain how to handle the split history when
  // reinserting nodes and reinserting nodes seems to hurt the performance, so
  // we don't do it.

  // We find the split axis that will be used if the topological split fails now
  // to save CPU time.

  // Find the next split axis.
  std::vector<bool> axes(tree->Bound().Dim());
  std::vector<int> dimensionsLastUsed(tree->NumChildren());
  for (size_t i = 0; i < tree->NumChildren(); i++)
    dimensionsLastUsed[i] = 
                    tree->Child(i).AuxiliaryInfo().SplitHistory().lastDimension;
  std::sort(dimensionsLastUsed.begin(), dimensionsLastUsed.end());

  size_t lastDim = dimensionsLastUsed[dimensionsLastUsed.size()/2];
  size_t minOverlapSplitDimension = tree->Bound().Dim();

  // See if we can use a new dimension.
  for (size_t i = lastDim + 1; i < axes.size(); i++)
  {
    axes[i] = true;
    for (size_t j = 0; j < tree->NumChildren(); j++)
      axes[i] = axes[i] & 
                tree->Child(j).AuxiliaryInfo().SplitHistory().history[i];
    if (axes[i] == true)
    {
      minOverlapSplitDimension = i;
      break;
    }
  }
  if (minOverlapSplitDimension == tree->Bound().Dim())
  {
    for (size_t i = 0; i < lastDim + 1; i++)
    {
      axes[i] = true;
      for (size_t j = 0; j < tree->NumChildren(); j++)
        axes[i] = axes[i] & 
                  tree->Child(j).AuxiliaryInfo().SplitHistory().history[i];
      if (axes[i] == true)
      {
        minOverlapSplitDimension = i;
        break;
      }
    }
  }

  bool minOverlapSplitUsesHi = false;
  ElemType bestScoreMinOverlapSplit = std::numeric_limits<ElemType>::max();
  ElemType areaOfBestMinOverlapSplit = 0;
  int bestIndexMinOverlapSplit = 0;

  int bestOverlapIndexOnBestAxis = 0;
  int bestAreaIndexOnBestAxis = 0;
  bool tiedOnOverlap = false;
  bool lowIsBest = true;
  int bestAxis = 0;
  ElemType bestAxisScore = std::numeric_limits<ElemType>::max();
  ElemType overlapBestOverlapAxis = 0;
  ElemType areaBestOverlapAxis = 0;
  ElemType overlapBestAreaAxis = 0;
  ElemType areaBestAreaAxis = 0;

  for (size_t j = 0; j < tree->Bound().Dim(); j++)
  {
    ElemType axisScore = 0.0;

    // We'll do Bound().Lo() now and use Bound().Hi() later.
    std::vector<std::pair<ElemType, size_t>> sorted(tree->NumChildren());
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[j].Lo();
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType>);

    // We'll store each of the three scores for each distribution.
    std::vector<ElemType> areas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> margins(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> overlapedAreas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    for (size_t i = 0; i < areas.size(); i++)
    {
      areas[i] = 0.0;
      margins[i] = 0.0;
      overlapedAreas[i] = 0.0;
    }

    for (size_t i = 0; i < areas.size(); i++)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinNumChildren() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.

      size_t cutOff = tree->MinNumChildren() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < cutOff; l++)
        bound1 |= tree->Child(sorted[l].second).Bound();

      for (size_t l = cutOff; l < tree->NumChildren(); l++)
        bound2 |= tree->Child(sorted[l].second).Bound();

      ElemType area1 = bound1.Volume();
      ElemType area2 = bound2.Volume();
      ElemType oArea = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); k++)
        margins[i] += bound1[k].Width() + bound2[k].Width();

      areas[i] += area1 + area2;
      overlapedAreas[i] += oArea;
      axisScore += margins[i];
    }

    if (axisScore < bestAxisScore)
    {
      bestAxisScore = axisScore;
      bestAxis = j;
      bestOverlapIndexOnBestAxis = 0;
      bestAreaIndexOnBestAxis = 0;
      overlapBestOverlapAxis = overlapedAreas[bestOverlapIndexOnBestAxis];
      areaBestOverlapAxis = areas[bestAreaIndexOnBestAxis];
      for (size_t i = 1; i < areas.size(); i++)
      {
        if (overlapedAreas[i] < overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = false;
          bestAreaIndexOnBestAxis = i;
          bestOverlapIndexOnBestAxis = i;
          overlapBestOverlapAxis = overlapedAreas[i];
          areaBestOverlapAxis = areas[i];
        }
        else if (overlapedAreas[i] ==
                 overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[bestAreaIndexOnBestAxis])
          {
            bestAreaIndexOnBestAxis = i;
            overlapBestAreaAxis = overlapedAreas[i];
            areaBestAreaAxis = areas[i];
          }
        }
      }
    }

    // Track the minOverlapSplit data
    if (minOverlapSplitDimension != tree->Bound().Dim() &&
        j == minOverlapSplitDimension)
    {
      for (size_t i = 0; i < overlapedAreas.size(); i++)
      {
        if (overlapedAreas[i] < bestScoreMinOverlapSplit)
        {
          bestScoreMinOverlapSplit = overlapedAreas[i];
          bestIndexMinOverlapSplit = i;
          areaOfBestMinOverlapSplit = areas[i];
        }
      }
    }
  }

  // Now we do the same thing using Bound().Hi() and choose the best of the two.
  for (size_t j = 0; j < tree->Bound().Dim(); j++)
  {
    ElemType axisScore = 0.0;

    std::vector<std::pair<ElemType, size_t>> sorted(tree->NumChildren());
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[j].Hi();
      sorted[i].second = i;
    }

    std::sort(sorted.begin(), sorted.end(), PairComp<ElemType>);

    // We'll store each of the three scores for each distribution.
    std::vector<ElemType> areas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> margins(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    std::vector<ElemType> overlapedAreas(tree->MaxNumChildren() -
        2 * tree->MinNumChildren() + 2);
    for (size_t i = 0; i < areas.size(); i++)
    {
      areas[i] = 0.0;
      margins[i] = 0.0;
      overlapedAreas[i] = 0.0;
    }

    for (size_t i = 0; i < areas.size(); i++)
    {
      // The ith arrangement is obtained by placing the first
      // tree->MinNumChildren() + i points in one rectangle and the rest in
      // another.  Then we calculate the three scores for that distribution.

      size_t cutOff = tree->MinNumChildren() + i;

      BoundType bound1(tree->Bound().Dim());
      BoundType bound2(tree->Bound().Dim());

      for (size_t l = 0; l < cutOff; l++)
        bound1 |= tree->Child(sorted[l].second).Bound();

      for (size_t l = cutOff; l < tree->NumChildren(); l++)
        bound2 |= tree->Child(sorted[l].second).Bound();

      ElemType area1 = bound1.Volume();
      ElemType area2 = bound2.Volume();
      ElemType oArea = bound1.Overlap(bound2);

      for (size_t k = 0; k < bound1.Dim(); k++)
        margins[i] += bound1[k].Width() + bound2[k].Width();


      areas[i] += area1 + area2;
      overlapedAreas[i] += oArea;
      axisScore += margins[i];
    }

    if (axisScore < bestAxisScore)
    {
      bestAxisScore = axisScore;
      bestAxis = j;
      lowIsBest = false;
      bestOverlapIndexOnBestAxis = 0;
      bestAreaIndexOnBestAxis = 0;
      overlapBestOverlapAxis = overlapedAreas[bestOverlapIndexOnBestAxis];
      areaBestOverlapAxis = areas[bestAreaIndexOnBestAxis];
      for (size_t i = 1; i < areas.size(); i++)
      {
        if (overlapedAreas[i] < overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = false;
          bestAreaIndexOnBestAxis = i;
          bestOverlapIndexOnBestAxis = i;
          overlapBestOverlapAxis = overlapedAreas[i];
          areaBestOverlapAxis = areas[i];
        }
        else if (overlapedAreas[i] ==
                 overlapedAreas[bestOverlapIndexOnBestAxis])
        {
          tiedOnOverlap = true;
          if (areas[i] < areas[bestAreaIndexOnBestAxis])
          {
            bestAreaIndexOnBestAxis = i;
            overlapBestAreaAxis = overlapedAreas[i];
            areaBestAreaAxis = areas[i];
          }
        }
      }
    }

    // Track the minOverlapSplit data
    if (minOverlapSplitDimension != tree->Bound().Dim() &&
        j == minOverlapSplitDimension)
    {
      for (size_t i = 0; i < overlapedAreas.size(); i++)
      {
        if (overlapedAreas[i] < bestScoreMinOverlapSplit)
        {
          minOverlapSplitUsesHi = true;
          bestScoreMinOverlapSplit = overlapedAreas[i];
          bestIndexMinOverlapSplit = i;
          areaOfBestMinOverlapSplit = areas[i];
        }
      }
    }
  }

  std::vector<std::pair<ElemType, size_t>> sorted(tree->NumChildren());
  if (lowIsBest)
  {
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[bestAxis].Lo();
      sorted[i].second = i;
    }
  }
  else
  {
    for (size_t i = 0; i < sorted.size(); i++)
    {
      sorted[i].first = tree->Child(i).Bound()[bestAxis].Hi();
      sorted[i].second = i;
    }
  }

  std::sort(sorted.begin(), sorted.end(), PairComp<ElemType>);

  TreeType* treeOne = new TreeType(tree->Parent(), tree->MaxNumChildren());
  TreeType* treeTwo = new TreeType(tree->Parent(), tree->MaxNumChildren());

  // Now as per the X-tree paper, we ensure that this split was good enough.
  bool useMinOverlapSplit = false;
  if (tiedOnOverlap)
  {
    if (overlapBestAreaAxis/areaBestAreaAxis < MAX_OVERLAP)
    {
      for (size_t i = 0; i < tree->NumChildren(); i++)
      {
        if (i < bestAreaIndexOnBestAxis + tree->MinNumChildren())
          InsertNodeIntoTree(treeOne, tree->children[sorted[i].second]);
        else
          InsertNodeIntoTree(treeTwo, tree->children[sorted[i].second]);
      }
    }
    else
      useMinOverlapSplit = true;
  }
  else
  {
    if (overlapBestOverlapAxis/areaBestOverlapAxis < MAX_OVERLAP)
    {
      for (size_t i = 0; i < tree->NumChildren(); i++)
      {
        if (i < bestOverlapIndexOnBestAxis + tree->MinNumChildren())
          InsertNodeIntoTree(treeOne, tree->children[sorted[i].second]);
        else
          InsertNodeIntoTree(treeTwo, tree->children[sorted[i].second]);
      }
    }
    else
      useMinOverlapSplit = true;
  }

  // If the split was not good enough, then we try the minimal overlap split.
  // If that fails, we create a "super node" (more accurately we resize this one
  // to make it a super node).
  if (useMinOverlapSplit)
  {
    // If there is a dimension that might work, try that.
    if ((minOverlapSplitDimension != tree->Bound().Dim()) &&
        (bestScoreMinOverlapSplit / areaOfBestMinOverlapSplit < MAX_OVERLAP))
    {
      std::vector<std::pair<ElemType, size_t>> sorted2(tree->NumChildren());
      if (minOverlapSplitUsesHi)
      {
        for (size_t i = 0; i < sorted2.size(); i++)
        {
          sorted2[i].first = tree->Child(i).Bound()[bestAxis].Hi();
          sorted2[i].second = i;
        }
      }
      else
      {
        for (size_t i = 0; i < sorted2.size(); i++)
        {
          sorted2[i].first = tree->Child(i).Bound()[bestAxis].Lo();
          sorted2[i].second = i;
        }
      }
      std::sort(sorted2.begin(), sorted2.end(), PairComp<ElemType>);

      for (size_t i = 0; i < tree->NumChildren(); i++)
      {
        if (i < bestIndexMinOverlapSplit + tree->MinNumChildren())
          InsertNodeIntoTree(treeOne, tree->children[sorted2[i].second]);
        else
          InsertNodeIntoTree(treeTwo, tree->children[sorted2[i].second]);
      }
    }
    else
    {
      // We don't create a supernode that would be the only child of the root.
      // (Note that if you did try to do so you would need to update the parent
      // field on each child of this new node as creating a supernode causes the
      // function to return before that is done.

      // I thought commenting out the bellow would make the tree less efficient
      // but would still work.  It doesn't.  I should look into that to see if
      // there is another bug.

      if ((tree->Parent()->Parent() == NULL) &&
          (tree->Parent()->NumChildren() == 1))
      {
        // We make the root a supernode instead.
        tree->Parent()->MaxNumChildren() = tree->MaxNumChildren() + 
                              tree->AuxiliaryInfo().NormalNodeMaxNumChildren();
        tree->Parent()->children.resize(tree->Parent()->MaxNumChildren() + 1);
        tree->Parent()->NumChildren() = tree->NumChildren();
        for (size_t i = 0; i < tree->NumChildren(); i++)
        {
          tree->Parent()->children[i] = tree->children[i];
          tree->Child(i).Parent() = tree->Parent();
        }

        delete treeOne;
        delete treeTwo;
        tree->NullifyData();
        tree->SoftDelete();
        return false;
      }

      // If we don't have to worry about the root, we just enlarge this node.
      tree->MaxNumChildren() += 
                              tree->AuxiliaryInfo().NormalNodeMaxNumChildren();
      tree->children.resize(tree->MaxNumChildren() + 1);
      for (size_t i = 0; i < tree->NumChildren(); i++)
        tree->Child(i).Parent() = tree;

      delete treeOne;
      delete treeTwo;

      return false;
    }
  }

  // Update the split history of each child.
  treeOne->AuxiliaryInfo().SplitHistory().history[bestAxis] = true;
  treeOne->AuxiliaryInfo().SplitHistory().lastDimension = bestAxis;
  treeTwo->AuxiliaryInfo().SplitHistory().history[bestAxis] = true;
  treeTwo->AuxiliaryInfo().SplitHistory().lastDimension = bestAxis;

  // Remove this node and insert treeOne and treeTwo
  TreeType* par = tree->Parent();
  size_t index = 0;
  for (size_t i = 0; i < par->NumChildren(); i++)
  {
    if (par->children[i] == tree)
    {
      index = i;
      break;
    }
  }

  par->children[index] = treeOne;
  par->children[par->NumChildren()++] = treeTwo;

  // we only add one at a time, so we should only need to test for equality
  // just in case, we use an assert.

  if (!(par->NumChildren() <= par->MaxNumChildren() + 1))
    Log::Debug << "error " << par->NumChildren() << ", "
        << par->MaxNumChildren() + 1 << std::endl;
  assert(par->NumChildren() <= par->MaxNumChildren() + 1);

  if (par->NumChildren() == par->MaxNumChildren() + 1)
    XTreeSplit::SplitNonLeafNode(par,relevels);

  // We have to update the children of each of these new nodes so that they
  // record the correct parent.
  for (size_t i = 0; i < treeOne->NumChildren(); i++)
    treeOne->Child(i).Parent() = treeOne;
  for (size_t i = 0; i < treeTwo->NumChildren(); i++)
    treeTwo->Child(i).Parent() = treeTwo;

  assert(treeOne->Parent()->NumChildren() <=
      treeOne->Parent()->MaxNumChildren());
  assert(treeOne->Parent()->NumChildren() >=
      treeOne->Parent()->MinNumChildren());
  assert(treeTwo->Parent()->NumChildren() <=
      treeTwo->Parent()->MaxNumChildren());
  assert(treeTwo->Parent()->NumChildren() >=
      treeTwo->Parent()->MinNumChildren());

  tree->SoftDelete();

  return false;
}

/**
 * Insert a node into another node.  Expanding the bounds and updating the
 * numberOfChildren.
 */
template<typename TreeType>
void XTreeSplit::InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->numDescendants += srcNode->numDescendants;
  destTree->children[destTree->NumChildren()] = srcNode;
  destTree->NumChildren()++;
}

} // namespace tree
} // namespace mlpack

#endif
