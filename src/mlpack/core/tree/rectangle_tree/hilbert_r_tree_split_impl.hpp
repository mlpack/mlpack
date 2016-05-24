/**
 * @file hilbert_r_tree_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (HilbertRTreeSplit) to split a RectangleTree.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP

#include "hilbert_r_tree_split.hpp"
#include "rectangle_tree.hpp"
#include <mlpack/core/math/range.hpp>

namespace mlpack {
namespace tree {

template<typename TreeType,typename HilbertValue>
HilbertRTreeSplit::HilbertRTreeSplit()
{
}

template<typename TreeType,typename HilbertValue>
HilbertRTreeSplit(const TreeType *node)
{
}

template<typename TreeType,typename HilbertValue>
HilbertRTreeSplit(const TreeType &other)
{
}

template<typename TreeType,typename HilbertValue>
void HilbertRTreeSplit<TreeType,HilbertValue>::SplitLeafNode(TreeType *tree,std::vector<bool>& relevels)
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
    copy->Split().SplitLeafNode(copy,relevels);
    return;
  }

  TreeType *parent = tree->Parent();

  size_t iTree = 0;
  for(iTree = 0;parent->Children()[iTree] != tree; iTree++);

  size_t firstSibling,lastSibling;
  if(FindCooperatingSiblings(parent,iTree,firstSibling,lastSibling))
  {
    RedistributePointsEvenly(parent,firstSibling,lastSibling);
    return;
  }


  size_t iNewSibling = (iTree + splitOrder < parent->NumChildren() ? iTree + splitOrder : parent->NumChildren());

  for(size_t i = parent->NumChildren(); i > iNewSibling ; i--)
    parent->Children()[i] = parent->Children[i-1];

  parent->NumChildren()++;

  parent->Children()[iNewSibling] = new TreeType(parent);

  lastSibling = (iTree + splitOrder < parent->NumChildren() ? iTree + splitOrder : parent->NumChildren() - 1);
  firstSibling = (lastSibling > splitOrder ? lastSibling - splitOrder : 0);

  RedistributePointsEvenly(parent,firstSibling,lastSibling);

  if(parent->NumChildren() == parent->MaxNumChildren() + 1)
    parent->Split().SplitNonLeafNode(parent,relevels);

}

template<typename TreeType,typename HilbertValue>
void HilbertRTreeSplit<TreeType,HilbertValue>::SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels)
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
    copy->Split().SplitLeafNode(copy,relevels);
    return;
  }

  TreeType *parent = tree->Parent();

  size_t iTree = 0;
  for(iTree = 0;parent->Children()[iTree] != tree; iTree++);

  size_t firstSibling,lastSibling;
  if(FindCooperatingSiblings(parent,iTree,firstSibling,lastSibling))
  {
    RedistributeNodesEvenly(parent,firstSibling,lastSibling);
    return;
  }

  size_t iNewSibling = (iTree + splitOrder < parent->NumChildren() ? iTree + splitOrder : parent->NumChildren());

  for(size_t i = parent->NumChildren(); i > iNewSibling ; i--)
    parent->Children()[i] = parent->Children[i-1];

  parent->NumChildren()++;

  parent->Children()[iNewSibling] = new TreeType(parent);

  lastSibling = (iTree + splitOrder < parent->NumChildren() ? iTree + splitOrder : parent->NumChildren() - 1);
  firstSibling = (lastSibling > splitOrder ? lastSibling - splitOrder : 0);

  RedistributeNodesEvenly(parent,firstSibling,lastSibling);

  if(parent->NumChildren() == parent->MaxNumChildren() + 1)
    parent->Split().SplitNonLeafNode(parent,relevels);
}

template<typename TreeType,typename HilbertValue>
bool HilbertRTreeSplit<TreeType,HilbertValue>::FindCooperatingSiblings(TreeType *parent,size_t iTree,size_t &firstSubling,size_t &lastSibling)
{
  size_t start = (iTree > splitOrder-1 ? iTree - splitOrder + 1 : 0);
  size_t end = (iTree + splitOrder <= parent->NumChildren() ? iTree + splitOrder : parent->NumChildren());

  size_t iUnderfullSibling;
  if(parent->Children()[iTree]->NumChildren() != 0)
  {
    for(iUnderfullSibling = start; iUnderfullSibling < end; iUnderfullSibling++)
      if(parent->Children()][iUnderfullSibling]->NumChildren() < parent->Children()][iUnderfullSibling]->MaxNumChildren() - 1)
        break;
  }
  else
  {
    for(iUnderfullSibling = start; iUnderfullSibling < end; iUnderfullSibling++)
      if(parent->Children()][iUnderfullSibling]->NumPoints() < parent->Children()][iUnderfullSibling]->MaxLeafSize() - 1)
        break;
  }

  if(iUnderfullSibling == end)
    return false;

  if(iUnderfullSibling > iTree)
  {
    lastSibling = (iTree + splitOrder-1 < parent->NumChildren() ? iTree + splitOrder-1 : parent->NumChildren() - 1);
    firstSibling = (lastSibling > splitOrder-1 ? lastSibling - splitOrder + 1 : 0);
  }
  else
  {
    lastSibling = (iUnderfullSibling + splitOrder-1 < parent->NumChildren() ? iUnderfullSibling + splitOrder-1 : parent->NumChildren() - 1);
    firstSibling = (lastSibling > splitOrder-1 ? lastSibling - splitOrder + 1 : 0);
  }

  return true;
}

template<typename TreeType,typename HilbertValue>
void HilbertRTreeSplit<TreeType,HilbertValue>::RedistributeNodesEvenly(const TreeType *parent,size_t firstSibling,size_t lastSibling)
{
  size_t numChildren = 0;
  size_t numChildrenPerNode,numRestChildren;

  for(size_t i = firstSibling; i <= lastSibling; i++)
    numChildren += parent->Children()[i]->NumChildren();

  numChildrenPerNode = numChildren / (lastSibling - firstSibling + 1);
  numRestChildren = numChildren % (lastSibling - firstSibling + 1);

  std::vector<TreeType*> children(numChildren);

  size_t iChild = 0;
  for(size_t i = firstSibling; i <= lastSibling; i++)
  {
    for(size_t j = 0; j < parent->Children()[i]->NumChildren(); j++)
    {
      children[iChild] = parent->Children()[i]->Children()[j];
      iChild++;
    }
  }

  iChild = 0;
  for(size_t i = firstSibling; i <= lastSibling; i++)
  {
    for(size_t j = 0; j < numChildrenPerNode; j++)
    {
      parent->Children()[i]->Children()[j] = children[iChild];
      children[iChild]->Parent() = parent->Children()[i];
      iChild++;
    }
    if(numRestChildren > 0)
    {
      parent->Children()[i]->Children()[numChildrenPerNode] = children[iChild];
      children[iChild]->Parent() = parent->Children()[i];
      parent->Children()[i]->NumChildren() = numChildrenPerNode + 1;
      numRestChildren--;
      iChild++;
    }
    else
    {
      parent->Children()[i]->NumChildren() = numChildrenPerNode;
    }
    parent->Children()[i]->Split().largestHilbertValue = children[iChild-1]->Split().largestHilbertValue;
  }
}

template<typename TreeType,typename HilbertValue>
void HilbertRTreeSplit<TreeType,HilbertValue>::RedistributePointsEvenly(const TreeType *parent,size_t firstSibling,size_t lastSibling)
{
  size_t numPoints = 0;
  size_t numPointsPerNode,numRestPoints;

  for(size_t i = firstSibling; i <= lastSibling; i++)
    numPoints += parent->Children()[i]->NumPoints();

  numPointsPerNode = numPoints / (lastSibling - firstSibling + 1);
  numRestPoints = numPoints % (lastSibling - firstSibling + 1);

  std::vector<size_t> points(numPoints);

  size_t iPoint = 0;
  for(size_t i = firstSibling; i <= lastSibling; i++)
  {
    for(size_t j = 0; j < parent->Children()[i]->NumPoints(); j++)
    {
      points[iPoint] = parent->Children()[i]->Points()[j];
      iPoint++;
    }
  }

  iPoint = 0;
  for(size_t i = firstSibling; i <= lastSibling; i++)
  {
    parent->Children()[i]->Bound().Clear();

    for(size_t j = 0; j < numPointsPerNode; j++)
    {
      parent->Children()[i]->Bound() |= parent->Children()[i]->Dataset()->col(points[iPoint]);
      parent->Children()[i]->Points()[j] = points[iPoint];
      parent->Children()[i]->LocalDataset()->col(j) = parent->Children()[i]->Dataset()->col(points[iPoint]);
      iPoint++;
    }
    if(numRestPoints > 0)
    {
      parent->Children()[i]->Bound() |= parent->Children()[i]->Dataset()->col(points[iPoint]);
      parent->Children()[i]->Points()[j] = points[iPoint];
      parent->Children()[i]->LocalDataset()->col(j) = parent->Children()[i]->Dataset()->col(points[iPoint]);
      parent->Children()[i]->NumPoints() = numPointsPerNode + 1;
      numRestPoints--;
      iPoint++;
    }
    else
    {
      parent->Children()[i]->NumPoints() = numPointsPerNode;
    }
//   TODO:
//    Adjust the largestHilbertValue
//    parent->Children()[i]->Split().largestHilbertValue.AdjustValue();
  }
}

/**
 * Serialize the split.
 */
template<typename TreeType,typename HilbertValue>
template<typename Archive>
void XTreeSplit<TreeType>::Serialize(Archive& ar,const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(largestHilbertValue, "largestHilbertValue");
}

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP
