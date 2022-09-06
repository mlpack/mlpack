/**
 * @file core/tree/rectangle_tree/hilbert_r_tree_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (HilbertRTreeSplit) to split a RectangleTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP

#include "hilbert_r_tree_split.hpp"
#include "rectangle_tree.hpp"
#include <mlpack/core/math/range.hpp>

namespace mlpack {

template<size_t splitOrder>
template<typename TreeType>
void HilbertRTreeSplit<splitOrder>::SplitLeafNode(TreeType* tree,
                                                  std::vector<bool>& relevels)
{
  if (tree->Count() <= tree->MaxLeafSize())
    return;
  // If we are splitting the root node, we need will do things differently so
  // that the constructor and other methods don't confuse the end user by giving
  // an address of another node.
  if (tree->Parent() == NULL)
  {
    // We actually want to copy this way.  Pointers and everything.
    TreeType* copy = new TreeType(*tree, false);
    // Only the root node owns this variable.
    copy->AuxiliaryInfo().HilbertValue().OwnsValueToInsert() = false;
    // Only leaf nodes own this variable.
    tree->AuxiliaryInfo().HilbertValue().OwnsLocalHilbertValues() = false;
    copy->Parent() = tree;
    tree->Count() = 0;
    tree->NullifyData();
    // Because this was a leaf node, numChildren must be 0.
    tree->children[(tree->NumChildren())++] = copy;
    SplitLeafNode(copy, relevels);
    return;
  }

  TreeType* parent = tree->Parent();
  size_t iTree = 0;
  for (iTree = 0; parent->children[iTree] != tree; iTree++) { }

  // Try to find splitOrder cooperating siblings in order to redistribute points
  // among them and avoid split.
  size_t firstSibling, lastSibling;
  if (FindCooperatingSiblings(parent, iTree, firstSibling, lastSibling))
  {
    RedistributePointsEvenly(parent, firstSibling, lastSibling);
    return;
  }

  // We can not find splitOrder cooperating siblings since they are all full.
  // We introduce new one instead.
  size_t iNewSibling = (iTree + splitOrder < parent->NumChildren() ?
                        iTree + splitOrder : parent->NumChildren());

  for (size_t i = parent->NumChildren(); i > iNewSibling ; i--)
    parent->children[i] = parent->children[i - 1];

  parent->NumChildren()++;

  parent->children[iNewSibling] = new TreeType(parent);

  lastSibling = (iTree + splitOrder < parent->NumChildren() ?
                 iTree + splitOrder : parent->NumChildren() - 1);
  firstSibling = (lastSibling > splitOrder ? lastSibling - splitOrder : 0);

  assert(lastSibling - firstSibling <= splitOrder);
  assert(lastSibling < parent->NumChildren());

  // Redistribute the points among (splitOrder + 1) cooperating siblings evenly.
  RedistributePointsEvenly(parent, firstSibling, lastSibling);

  if (parent->NumChildren() == parent->MaxNumChildren() + 1)
    SplitNonLeafNode(parent, relevels);
}

template<size_t splitOrder>
template<typename TreeType>
bool HilbertRTreeSplit<splitOrder>::
SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels)
{
  // If we are splitting the root node, we need will do things differently so
  // that the constructor and other methods don't confuse the end user by giving
  // an address of another node.
  if (tree->Parent() == NULL)
  {
    // We actually want to copy this way.  Pointers and everything.
    TreeType* copy = new TreeType(*tree, false);
    // Only the root node owns this variable.
    copy->AuxiliaryInfo().HilbertValue().OwnsValueToInsert() = false;
    copy->Parent() = tree;
    tree->NumChildren() = 0;
    tree->NullifyData();
    tree->children[(tree->NumChildren())++] = copy;

    SplitNonLeafNode(copy, relevels);
    return true;
  }

  TreeType* parent = tree->Parent();

  size_t iTree = 0;
  for (iTree = 0; parent->children[iTree] != tree; iTree++) { }

  // Try to find splitOrder cooperating siblings in order to redistribute
  // children among them and avoid split.
  size_t firstSibling, lastSibling;
  if (FindCooperatingSiblings(parent, iTree, firstSibling, lastSibling))
  {
    RedistributeNodesEvenly(parent, firstSibling, lastSibling);
    return false;
  }

  // We can not find splitOrder cooperating siblings since they are all full.
  // We introduce new one instead.
  size_t iNewSibling = (iTree + splitOrder < parent->NumChildren() ?
                        iTree + splitOrder : parent->NumChildren());

  for (size_t i = parent->NumChildren(); i > iNewSibling ; i--)
    parent->children[i] = parent->children[i - 1];

  parent->NumChildren()++;

  parent->children[iNewSibling] = new TreeType(parent);

  lastSibling = (iTree + splitOrder < parent->NumChildren() ?
                 iTree + splitOrder : parent->NumChildren() - 1);
  firstSibling = (lastSibling > splitOrder ?
                  lastSibling - splitOrder : 0);

  assert(lastSibling - firstSibling <= splitOrder);
  assert(lastSibling < parent->NumChildren());

  // Redistribute children among (splitOrder + 1) cooperating siblings evenly.
  RedistributeNodesEvenly(parent, firstSibling, lastSibling);

  if (parent->NumChildren() == parent->MaxNumChildren() + 1)
    SplitNonLeafNode(parent, relevels);
  return false;
}

template<size_t splitOrder>
template<typename TreeType>
bool HilbertRTreeSplit<splitOrder>::FindCooperatingSiblings(
    TreeType* parent,
    const size_t iTree,
    size_t& firstSibling,
    size_t& lastSibling)
{
  size_t start = (iTree > splitOrder - 1 ? iTree - splitOrder + 1 : 0);
  size_t end = (iTree + splitOrder <= parent->NumChildren() ?
                iTree + splitOrder : parent->NumChildren());

  size_t iUnderfullSibling;

  // Try to find empty space among cooperating siblings.
  if (parent->Child(iTree).NumChildren() != 0)
  {
    for (iUnderfullSibling = start; iUnderfullSibling < end;
         iUnderfullSibling++)
      if (parent->Child(iUnderfullSibling).NumChildren() <
          parent->Child(iUnderfullSibling).MaxNumChildren() - 1)
        break;
  }
  else
  {
    for (iUnderfullSibling = start; iUnderfullSibling < end;
         iUnderfullSibling++)
      if (parent->Child(iUnderfullSibling).NumPoints() <
          parent->Child(iUnderfullSibling).MaxLeafSize() - 1)
        break;
  }

  if (iUnderfullSibling == end) // All nodes are full.
    return false;

  if (iUnderfullSibling > iTree)
  {
    lastSibling = (iTree + splitOrder - 1 < parent->NumChildren() ?
                   iTree + splitOrder - 1 : parent->NumChildren() - 1);
    firstSibling = (lastSibling > splitOrder - 1 ?
                    lastSibling - splitOrder + 1 : 0);
  }
  else
  {
    lastSibling = (iUnderfullSibling + splitOrder - 1 < parent->NumChildren() ?
        iUnderfullSibling + splitOrder - 1 : parent->NumChildren() - 1);
    firstSibling = (lastSibling > splitOrder - 1 ?
        lastSibling - splitOrder + 1 : 0);
  }

  assert(lastSibling - firstSibling <= splitOrder - 1);
  assert(lastSibling < parent->NumChildren());

  return true;
}

template<size_t splitOrder>
template<typename TreeType>
void HilbertRTreeSplit<splitOrder>::
RedistributeNodesEvenly(const TreeType *parent,
                        size_t firstSibling, size_t lastSibling)
{
  size_t numChildren = 0;
  size_t numChildrenPerNode, numRestChildren;

  for (size_t i = firstSibling; i <= lastSibling; ++i)
    numChildren += parent->Child(i).NumChildren();

  numChildrenPerNode = numChildren / (lastSibling - firstSibling + 1);
  numRestChildren = numChildren % (lastSibling - firstSibling + 1);

  std::vector<TreeType*> children(numChildren);

  // Copy children's children in order to redistribute them.
  size_t iChild = 0;
  for (size_t i = firstSibling; i <= lastSibling; ++i)
  {
    for (size_t j = 0; j < parent->Child(i).NumChildren(); ++j)
    {
      children[iChild] = parent->Child(i).children[j];
      iChild++;
    }
  }

  iChild = 0;
  for (size_t i = firstSibling; i <= lastSibling; ++i)
  {
    // Since we redistribute children of a sibling we should recalculate the
    // bound.
    parent->Child(i).Bound().Clear();
    parent->Child(i).numDescendants = 0;

    for (size_t j = 0; j < numChildrenPerNode; ++j)
    {
      parent->Child(i).Bound() |= children[iChild]->Bound();
      parent->Child(i).numDescendants += children[iChild]->numDescendants;
      parent->Child(i).children[j] = children[iChild];
      children[iChild]->Parent() = parent->children[i];
      iChild++;
    }
    if (numRestChildren > 0)
    {
      parent->Child(i).Bound() |= children[iChild]->Bound();
      parent->Child(i).numDescendants += children[iChild]->numDescendants;
      parent->Child(i).children[numChildrenPerNode] = children[iChild];
      children[iChild]->Parent() = parent->children[i];
      parent->Child(i).NumChildren() = numChildrenPerNode + 1;
      numRestChildren--;
      iChild++;
    }
    else
    {
      parent->Child(i).NumChildren() = numChildrenPerNode;
    }
    assert(parent->Child(i).NumChildren() <=
           parent->Child(i).MaxNumChildren());

    // Fix the largest Hilbert value of the sibling.
    parent->Child(i).AuxiliaryInfo().HilbertValue().UpdateLargestValue(
        parent->children[i]);
  }
}

template<size_t splitOrder>
template<typename TreeType>
void HilbertRTreeSplit<splitOrder>::
RedistributePointsEvenly(TreeType* parent,
                         const size_t firstSibling,
                         const size_t lastSibling)
{
  size_t numPoints = 0;
  size_t numPointsPerNode, numRestPoints;

  for (size_t i = firstSibling; i <= lastSibling; ++i)
    numPoints += parent->Child(i).NumPoints();

  numPointsPerNode = numPoints / (lastSibling - firstSibling + 1);
  numRestPoints = numPoints % (lastSibling - firstSibling + 1);

  std::vector<size_t> points(numPoints);

  // Copy children's points in order to redistribute them.
  size_t iPoint = 0;
  for (size_t i = firstSibling; i <= lastSibling; ++i)
  {
    for (size_t j = 0; j < parent->Child(i).NumPoints(); ++j)
      points[iPoint++] = parent->Child(i).Point(j);
  }

  iPoint = 0;
  for (size_t i = firstSibling; i <= lastSibling; ++i)
  {
    // Since we redistribute points of a sibling we should recalculate the
    // bound.
    parent->Child(i).Bound().Clear();

    size_t j;
    for (j = 0; j < numPointsPerNode; ++j)
    {
      parent->Child(i).Bound() |= parent->Dataset().col(points[iPoint]);
      parent->Child(i).Point(j) = points[iPoint];
      iPoint++;
    }
    if (numRestPoints > 0)
    {
      parent->Child(i).Bound() |= parent->Dataset().col(points[iPoint]);
      parent->Child(i).Point(j) = points[iPoint];
      parent->Child(i).Count() = numPointsPerNode + 1;
      numRestPoints--;
      iPoint++;
    }
    else
    {
      parent->Child(i).Count() = numPointsPerNode;
    }
    parent->Child(i).numDescendants = parent->Child(i).Count();

    assert(parent->Child(i).NumPoints() <=
           parent->Child(i).MaxLeafSize());
  }

  // Fix the largest Hilbert values of the siblings.
  parent->AuxiliaryInfo().HilbertValue().RedistributeHilbertValues(parent,
      firstSibling, lastSibling);

  TreeType* root = parent;

  while (root != NULL)
  {
    root->AuxiliaryInfo().HilbertValue().UpdateLargestValue(root);
    root = root->Parent();
  }
}

} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP
