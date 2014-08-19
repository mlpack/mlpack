/**
 * @file r_tree_split_impl.hpp
 * @author Andrew Wells
 *
 * Implementation of class (RTreeSplit) to split a RectangleTree.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_IMPL_HPP

#include "r_tree_split.hpp"
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
template<typename DescentType,
typename StatisticType,
typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::SplitLeafNode(
    TreeType* tree,
    std::vector<bool>& relevels)
{
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
    tree->Children()[(tree->NumChildren())++] = copy;
    SplitLeafNode(copy, relevels);
    return;
  }

  assert(tree->Parent()->NumChildren() <= tree->Parent()->MaxNumChildren());

  // Use the quadratic split method from: Guttman "R-Trees: A Dynamic Index
  // Structure for Spatial Searching"  It is simplified since we don't handle
  // rectangles, only points.  We assume that the tree uses Euclidean Distance.
  int i = 0;
  int j = 0;
  GetPointSeeds(*tree, i, j);

  TreeType* treeOne = new TreeType(tree->Parent());
  TreeType* treeTwo = new TreeType(tree->Parent());

  // This will assign the ith and jth point appropriately.
  AssignPointDestNode(tree, treeOne, treeTwo, i, j);

  // Remove this node and insert treeOne and treeTwo.
  TreeType* par = tree->Parent();
  size_t index = 0;
  while (par->Children()[index] != tree) { ++index; }

  par->Children()[index] = treeOne;
  par->Children()[par->NumChildren()++] = treeTwo;

  // We only add one at a time, so we should only need to test for equality
  // just in case, we use an assert.
  assert(par->NumChildren() <= par->MaxNumChildren() + 1);
  if (par->NumChildren() == par->MaxNumChildren() + 1)
    SplitNonLeafNode(par, relevels);

  assert(treeOne->Parent()->NumChildren() <= treeOne->MaxNumChildren());
  assert(treeOne->Parent()->NumChildren() >= treeOne->MinNumChildren());
  assert(treeTwo->Parent()->NumChildren() <= treeTwo->MaxNumChildren());
  assert(treeTwo->Parent()->NumChildren() >= treeTwo->MinNumChildren());

  // We need to delete this carefully since references to points are used.
  tree->SoftDelete();
}

/**
 * We call GetBoundSeeds to get the two new nodes that this one will be broken
 * into.  Then we call AssignNodeDestNode to move the children of this node into
 * either of those two nodes.  Finally, we delete the now unused information and
 * recurse up the tree if necessary.  We don't need to worry about the bounds
 * higher up the tree because they were already updated if necessary.
 */
template<typename DescentType,
         typename StatisticType,
         typename MatType>
bool RTreeSplit<DescentType, StatisticType, MatType>::SplitNonLeafNode(
    TreeType* tree,
    std::vector<bool>& relevels)
{
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
    tree->Children()[(tree->NumChildren())++] = copy;
    SplitNonLeafNode(copy, relevels);
    return true;
  }

  int i = 0;
  int j = 0;
  GetBoundSeeds(*tree, i, j);

  assert(i != j);

  TreeType* treeOne = new TreeType(tree->Parent());
  TreeType* treeTwo = new TreeType(tree->Parent());

  // This will assign the ith and jth rectangles appropriately.
  AssignNodeDestNode(tree, treeOne, treeTwo, i, j);

  // Remove this node and insert treeOne and treeTwo.
  TreeType* par = tree->Parent();
  size_t index = 0;
  while (par->Children()[index] != tree) { ++index; }

  assert(index != -1);
  par->Children()[index] = treeOne;
  par->Children()[par->NumChildren()++] = treeTwo;

  for (int i = 0; i < par->NumChildren(); i++)
    assert(par->Children()[i] != tree);

  // We only add one at a time, so should only need to test for equality just in
  // case, we use an assert.
  assert(par->NumChildren() <= par->MaxNumChildren() + 1);

  if (par->NumChildren() == par->MaxNumChildren() + 1)
    SplitNonLeafNode(par, relevels);

  // We have to update the children of each of these new nodes so that they
  // record the correct parent.
  for (int i = 0; i < treeOne->NumChildren(); i++)
    treeOne->Children()[i]->Parent() = treeOne;

  for (int i = 0; i < treeTwo->NumChildren(); i++)
    treeTwo->Children()[i]->Parent() = treeTwo;

  assert(treeOne->NumChildren() <= treeOne->MaxNumChildren());
  assert(treeTwo->NumChildren() <= treeTwo->MaxNumChildren());
  assert(treeOne->Parent()->NumChildren() <= treeOne->MaxNumChildren());

  // Because we now have pointers to the information stored under this tree,
  // we need to delete this node carefully.
  tree->SoftDelete(); //currently does nothing but leak memory.

  return false;
}

/**
 * Get the two points that will be used as seeds for the split of a leaf node.
 * The indices of these points will be stored in iRet and jRet.
 */
template<typename DescentType,
         typename StatisticType,
         typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::GetPointSeeds(
    const TreeType& tree,
    int& iRet,
    int& jRet)
{
  // Here we want to find the pair of points that it is worst to place in the
  // same node.  Because we are just using points, we will simply choose the two
  // that would create the most voluminous hyperrectangle.
  double worstPairScore = -1.0;
  for (int i = 0; i < tree.Count(); i++)
  {
    for (int j = i + 1; j < tree.Count(); j++)
    {
      const double score = arma::prod(arma::abs(tree.LocalDataset().col(i) -
          tree.LocalDataset().col(j)));

      if (score > worstPairScore)
      {
        worstPairScore = score;
        iRet = i;
        jRet = j;
      }
    }
  }
}

/**
 * Get the two bounds that will be used as seeds for the split of the node.  The
 * indices of the bounds will be stored in iRet and jRet.
 */
template<typename DescentType,
         typename StatisticType,
         typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::GetBoundSeeds(
    const TreeType& tree,
    int& iRet,
    int& jRet)
{
  double worstPairScore = -1.0;
  for (int i = 0; i < tree.NumChildren(); i++)
  {
    for (int j = i + 1; j < tree.NumChildren(); j++)
    {
      double score = 1.0;
      for (int k = 0; k < tree.Bound().Dim(); k++)
      {
        const double hiMax = std::max(tree.Children()[i]->Bound()[k].Hi(),
                                      tree.Children()[j]->Bound()[k].Hi());
        const double loMin = std::min(tree.Children()[i]->Bound()[k].Lo(),
                                      tree.Children()[j]->Bound()[k].Lo());
        score *= (hiMax - loMin);
      }

      if (score > worstPairScore)
      {
        worstPairScore = score;
        iRet = i;
        jRet = j;
      }
    }
  }
}

template<typename DescentType,
         typename StatisticType,
         typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::AssignPointDestNode(
    TreeType* oldTree,
    TreeType* treeOne,
    TreeType* treeTwo,
    const int intI,
    const int intJ)
{
  int end = oldTree->Count();

  assert(end > 1); // If this isn't true, the tree is really weird.

  // Restart the point counts since we are going to move them.
  oldTree->Count() = 0;
  treeOne->Count() = 0;
  treeTwo->Count() = 0;

  treeOne->InsertPoint(oldTree->Points()[intI]);
  treeTwo->InsertPoint(oldTree->Points()[intJ]);

  // If intJ is the last point in the tree, we need to switch the order so that
  // we remove the correct points.
  if (intI > intJ)
  {
    oldTree->Points()[intI] = oldTree->Points()[--end]; // Decrement end.
    oldTree->LocalDataset().col(intI) = oldTree->LocalDataset().col(end);
    oldTree->Points()[intJ] = oldTree->Points()[--end]; // Decrement end.
    oldTree->LocalDataset().col(intJ) = oldTree->LocalDataset().col(end);
  }
  else
  {
    oldTree->Points()[intJ] = oldTree->Points()[--end]; // Decrement end.
    oldTree->LocalDataset().col(intJ) = oldTree->LocalDataset().col(end);
    oldTree->Points()[intI] = oldTree->Points()[--end]; // Decrement end.
    oldTree->LocalDataset().col(intI) = oldTree->LocalDataset().col(end);
  }

  int numAssignedOne = 1;
  int numAssignedTwo = 1;

  // In each iteration, we go through all points and find the one that causes
  // the least increase of volume when added to one of the rectangles.  We then
  // add it to that rectangle.  We stop when we run out of points or when all of
  // the remaining points need to be assigned to the same rectangle to satisfy
  // the minimum fill requirement.

  // The below is safe because if end decreases and the right hand side of the
  // second part of the conjunction changes on the same iteration, we added the
  // point to the node with fewer points anyways.
  while ((end > 0) && (end > oldTree->MinLeafSize() -
      std::min(numAssignedOne, numAssignedTwo)))
  {
    int bestIndex = 0;
    double bestScore = DBL_MAX;
    int bestRect = 1;

    // Calculate the increase in volume for assigning this point to each
    // rectangle.

    // First, calculate the starting volume.
    double volOne = 1.0;
    double volTwo = 1.0;
    for (int i = 0; i < oldTree->Bound().Dim(); i++)
    {
      volOne *= treeOne->Bound()[i].Width();
      volTwo *= treeTwo->Bound()[i].Width();
    }

    // Find the point that, when assigned to one of the two new rectangles,
    // minimizes the increase in volume.
    for (int index = 0; index < end; index++)
    {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for (int i = 0; i < oldTree->Bound().Dim(); i++)
      {
        double c = oldTree->LocalDataset().col(index)[i];
        newVolOne *= treeOne->Bound()[i].Contains(c) ?
            treeOne->Bound()[i].Width() : (c < treeOne->Bound()[i].Lo() ?
            (treeOne->Bound()[i].Hi() - c) : (c - treeOne->Bound()[i].Lo()));
        newVolTwo *= treeTwo->Bound()[i].Contains(c) ?
            treeTwo->Bound()[i].Width() : (c < treeTwo->Bound()[i].Lo() ?
            (treeTwo->Bound()[i].Hi() - c) : (c - treeTwo->Bound()[i].Lo()));
      }

      // Choose the rectangle that requires the lesser increase in volume.
      if ((newVolOne - volOne) < (newVolTwo - volTwo))
      {
        if (newVolOne - volOne < bestScore)
        {
          bestScore = newVolOne - volOne;
          bestIndex = index;
          bestRect = 1;
        }
      }
      else
      {
        if (newVolTwo - volTwo < bestScore)
        {
          bestScore = newVolTwo - volTwo;
          bestIndex = index;
          bestRect = 2;
        }
      }
    }

    // Assign the point that causes the least increase in volume
    // to the appropriate rectangle.
    if (bestRect == 1)
    {
      treeOne->InsertPoint(oldTree->Points()[bestIndex]);
      numAssignedOne++;
    }
    else
    {
      treeTwo->InsertPoint(oldTree->Points()[bestIndex]);
      numAssignedTwo++;
    }

    oldTree->Points()[bestIndex] = oldTree->Points()[--end]; // Decrement end.
    oldTree->LocalDataset().col(bestIndex) = oldTree->LocalDataset().col(end);
  }

  // See if we need to satisfy the minimum fill.
  if (end > 0)
  {
    if (numAssignedOne < numAssignedTwo)
    {
      for (int i = 0; i < end; i++)
        treeOne->InsertPoint(oldTree->Points()[i]);
    }
    else
    {
      for (int i = 0; i < end; i++)
        treeTwo->InsertPoint(oldTree->Points()[i]);
    }
  }
}

template<typename DescentType,
         typename StatisticType,
         typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::AssignNodeDestNode(
    TreeType* oldTree,
    TreeType* treeOne,
    TreeType* treeTwo,
    const int intI,
    const int intJ)
{

  int end = oldTree->NumChildren();
  assert(end > 1); // If this isn't true, the tree is really weird.

  assert(intI != intJ);

  for (int i = 0; i < oldTree->NumChildren(); i++)
    for (int j = i + 1; j < oldTree->NumChildren(); j++)
      assert(oldTree->Children()[i] != oldTree->Children()[j]);

  InsertNodeIntoTree(treeOne, oldTree->Children()[intI]);
  InsertNodeIntoTree(treeTwo, oldTree->Children()[intJ]);

  // If intJ is the last node in the tree, we need to switch the order so that
  // we remove the correct nodes.
  if (intI > intJ)
  {
    oldTree->Children()[intI] = oldTree->Children()[--end];
    oldTree->Children()[intJ] = oldTree->Children()[--end];
  }
  else
  {
    oldTree->Children()[intJ] = oldTree->Children()[--end];
    oldTree->Children()[intI] = oldTree->Children()[--end];
  }

  assert(treeOne->NumChildren() == 1);
  assert(treeTwo->NumChildren() == 1);

  for (int i = 0; i < end; i++)
    for (int j = i + 1; j < end; j++)
      assert(oldTree->Children()[i] != oldTree->Children()[j]);

  for (int i = 0; i < end; i++)
    assert(oldTree->Children()[i] != treeOne->Children()[0]);

  for (int i = 0; i < end; i++)
    assert(oldTree->Children()[i] != treeTwo->Children()[0]);

  int numAssignTreeOne = 1;
  int numAssignTreeTwo = 1;

  // In each iteration, we go through all of the nodes and find the one that
  // causes the least increase of volume when added to one of the two new
  // rectangles.  We then add it to that rectangle.
  while ((end > 0) && (end > oldTree->MinNumChildren() -
      std::min(numAssignTreeOne, numAssignTreeTwo)))
  {
    int bestIndex = 0;
    double bestScore = DBL_MAX;
    int bestRect = 0;

    // Calculate the increase in volume for assigning this node to each of the
    // new rectangles.
    double volOne = 1.0;
    double volTwo = 1.0;
    for (int i = 0; i < oldTree->Bound().Dim(); i++)
    {
      volOne *= treeOne->Bound()[i].Width();
      volTwo *= treeTwo->Bound()[i].Width();
    }

    for (int index = 0; index < end; index++)
    {
      double newVolOne = 1.0;
      double newVolTwo = 1.0;
      for (int i = 0; i < oldTree->Bound().Dim(); i++)
      {
        // For each of the new rectangles, find the width in this dimension if
        // we add the rectangle at index to the new rectangle.
        const math::Range& range = oldTree->Children()[index]->Bound()[i];
        newVolOne *= treeOne->Bound()[i].Contains(range) ?
            treeOne->Bound()[i].Width() : (range.Contains(treeOne->Bound()[i]) ?
            range.Width() : (range.Lo() < treeOne->Bound()[i].Lo() ?
            (treeOne->Bound()[i].Hi() - range.Lo()) : (range.Hi() -
            treeOne->Bound()[i].Lo())));

        newVolTwo *= treeTwo->Bound()[i].Contains(range) ?
            treeTwo->Bound()[i].Width() : (range.Contains(treeTwo->Bound()[i]) ?
            range.Width() : (range.Lo() < treeTwo->Bound()[i].Lo() ?
            (treeTwo->Bound()[i].Hi() - range.Lo()) : (range.Hi() -
            treeTwo->Bound()[i].Lo())));
      }

      // Choose the rectangle that requires the lesser increase in volume.
      if ((newVolOne - volOne) < (newVolTwo - volTwo))
      {
        if (newVolOne - volOne < bestScore)
        {
          bestScore = newVolOne - volOne;
          bestIndex = index;
          bestRect = 1;
        }
      }
      else
      {
        if (newVolTwo - volTwo < bestScore)
        {
          bestScore = newVolTwo - volTwo;
          bestIndex = index;
          bestRect = 2;
        }
      }
    }

    // Assign the rectangle that causes the least increase in volume
    // to the appropriate rectangle.
    if (bestRect == 1)
    {
      InsertNodeIntoTree(treeOne, oldTree->Children()[bestIndex]);
      numAssignTreeOne++;
    }
    else
    {
      InsertNodeIntoTree(treeTwo, oldTree->Children()[bestIndex]);
      numAssignTreeTwo++;
    }

    oldTree->Children()[bestIndex] = oldTree->Children()[--end];
  }

  // See if we need to satisfy the minimum fill.
  if (end > 0)
  {
    if (numAssignTreeOne < numAssignTreeTwo)
    {
      for (int i = 0; i < end; i++)
      {
        InsertNodeIntoTree(treeOne, oldTree->Children()[i]);
        numAssignTreeOne++;
      }
    }
    else
    {
      for (int i = 0; i < end; i++)
      {
        InsertNodeIntoTree(treeTwo, oldTree->Children()[i]);
        numAssignTreeTwo++;
      }
    }
  }

  for (int i = 0; i < treeOne->NumChildren(); i++)
    for (int j = i + 1; j < treeOne->NumChildren(); j++)
      assert(treeOne->Children()[i] != treeOne->Children()[j]);

  for (int i = 0; i < treeTwo->NumChildren(); i++)
    for (int j = i + 1; j < treeTwo->NumChildren(); j++)
      assert(treeTwo->Children()[i] != treeTwo->Children()[j]);
}

/**
 * Insert a node into another node.  Expanding the bounds and updating the
 * numberOfChildren.
 */
template<typename DescentType,
         typename StatisticType,
         typename MatType>
void RTreeSplit<DescentType, StatisticType, MatType>::InsertNodeIntoTree(
    TreeType* destTree, TreeType* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->Children()[destTree->NumChildren()++] = srcNode;
}


}; // namespace tree
}; // namespace mlpack

#endif
