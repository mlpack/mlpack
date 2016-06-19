/**
 * @file r_plus_tree_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (RPlusTreeSplit) to split a RectangleTree.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_IMPL_HPP

#include "r_plus_tree_split.hpp"
#include "rectangle_tree.hpp"
#include "r_plus_plus_tree_auxiliary_information.hpp"
#include "r_plus_tree_split_policy.hpp"
#include "r_plus_plus_tree_split_policy.hpp"

namespace mlpack {
namespace tree {

template<typename SplitPolicyType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType>::SplitLeafNode(TreeType* tree, std::vector<bool>& relevels)
{
  if (tree->Count() == 1)
  {
    TreeType* node = tree->Parent();

    while (node != NULL)
    {
      if (node->NumChildren() == node->MaxNumChildren() + 1)
      {
        RPlusTreeSplit::SplitNonLeafNode(node,relevels);
        return;
      }
      node = node->Parent();
    }
    return;
  }
  else if (tree->Count() <= tree->MaxLeafSize())
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
    tree->Children()[(tree->NumChildren())++] = copy;
    assert(tree->NumChildren() == 1);

    RPlusTreeSplit::SplitLeafNode(copy,relevels);
    return;
  }

  const size_t fillFactor = tree->MaxLeafSize() * fillFactorFraction;
  size_t cutAxis;
  double cut;

  if ( !PartitionNode(tree, fillFactor, cutAxis, cut))
    return;

  assert(cutAxis < tree->Bound().Dim());

  TreeType* treeOne = new TreeType(tree->Parent());
  TreeType* treeTwo = new TreeType(tree->Parent());
  treeOne->MinLeafSize() = 0;
  treeOne->MinNumChildren() = 0;
  treeTwo->MinLeafSize() = 0;
  treeTwo->MinNumChildren() = 0;

  SplitLeafNodeAlongPartition(tree, treeOne, treeTwo, cutAxis, cut);

  TreeType* parent = tree->Parent();
  size_t i = 0;
  while (parent->Children()[i] != tree)
    i++;

  assert(i < parent->NumChildren());

  parent->Children()[i] = parent->Children()[--parent->NumChildren()];

  InsertNodeIntoTree(parent, treeOne);
  InsertNodeIntoTree(parent, treeTwo);

  assert(parent->NumChildren() <= parent->MaxNumChildren() + 1);
  if (parent->NumChildren() == parent->MaxNumChildren() + 1)
    RPlusTreeSplit::SplitNonLeafNode(parent, relevels);

  tree->SoftDelete();
}

template<typename SplitPolicyType>
template<typename TreeType>
bool RPlusTreeSplit<SplitPolicyType>::SplitNonLeafNode(TreeType* tree,
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

    RPlusTreeSplit::SplitNonLeafNode(copy,relevels);
    return true;
  }
  const size_t fillFactor = tree->MaxNumChildren() * fillFactorFraction;
  size_t cutAxis;
  double cut;

  if ( !PartitionNode(tree, fillFactor, cutAxis, cut))
    return false;

  assert(cutAxis < tree->Bound().Dim());

  TreeType* treeOne = new TreeType(tree->Parent());
  TreeType* treeTwo = new TreeType(tree->Parent());
  treeOne->MinLeafSize() = 0;
  treeOne->MinNumChildren() = 0;
  treeTwo->MinLeafSize() = 0;
  treeTwo->MinNumChildren() = 0;

  SplitNonLeafNodeAlongPartition(tree, treeOne, treeTwo, cutAxis, cut);

  TreeType* parent = tree->Parent();
  size_t i = 0;
  while (parent->Children()[i] != tree)
    i++;

  assert(i < parent->NumChildren());

  parent->Children()[i] = parent->Children()[--parent->NumChildren()];

  InsertNodeIntoTree(parent, treeOne);
  InsertNodeIntoTree(parent, treeTwo);

  tree->SoftDelete();

  assert(parent->NumChildren() <= parent->MaxNumChildren() + 1);
  
  if (parent->NumChildren() == parent->MaxNumChildren() + 1)
    RPlusTreeSplit::SplitNonLeafNode(parent, relevels);

  return false;
}

template<typename SplitPolicyType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType>::SplitLeafNodeAlongPartition(TreeType* tree,
  TreeType* treeOne, TreeType* treeTwo, size_t cutAxis, double cut)
{
  tree->AuxiliaryInfo().SplitAuxiliaryInfo(treeOne, treeTwo, cutAxis, cut);

  for (size_t i = 0; i < tree->NumPoints(); i++)
  {
    if (tree->Dataset().col(tree->Point(i))[cutAxis] <= cut)
    {
      treeOne->Points()[treeOne->Count()++] = tree->Point(i);
      treeOne->Bound() |= tree->Dataset().col(tree->Point(i));
    }
    else
    {
      treeTwo->Points()[treeTwo->Count()++] = tree->Point(i);
      treeTwo->Bound() |= tree->Dataset().col(tree->Point(i));
    }
  }
  assert(treeOne->Count() <= treeOne->MaxLeafSize());
  assert(treeTwo->Count() <= treeTwo->MaxLeafSize());
  
  assert(tree->Count() == treeOne->Count() + treeTwo->Count());
  assert(treeOne->Bound()[cutAxis].Hi() < treeTwo->Bound()[cutAxis].Lo());
}

template<typename SplitPolicyType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType>::SplitNonLeafNodeAlongPartition(TreeType* tree,
  TreeType* treeOne, TreeType* treeTwo, size_t cutAxis, double cut)
{
  tree->AuxiliaryInfo().SplitAuxiliaryInfo(treeOne, treeTwo, cutAxis, cut);

  for (size_t i = 0; i < tree->NumChildren(); i++)
  {
    TreeType* child = tree->Children()[i];
    int policy = SplitPolicyType::GetSplitPolicy(child, cutAxis, cut);

    if (policy == SplitPolicyType::AssignToFirstTree)
    {
      InsertNodeIntoTree(treeOne, child);
      child->Parent() = treeOne;
    }
    else if (policy == SplitPolicyType::AssignToSecondTree)
    {
      InsertNodeIntoTree(treeTwo, child);
      child->Parent() = treeTwo;
    }
    else
    {
      TreeType* childOne = new TreeType(treeOne);
      TreeType* childTwo = new TreeType(treeTwo);
      treeOne->MinLeafSize() = 0;
      treeOne->MinNumChildren() = 0;
      treeTwo->MinLeafSize() = 0;
      treeTwo->MinNumChildren() = 0;

      if (child->IsLeaf())
        SplitLeafNodeAlongPartition(child, childOne, childTwo, cutAxis, cut);
      else
        SplitNonLeafNodeAlongPartition(child, childOne, childTwo, cutAxis, cut);

      InsertNodeIntoTree(treeOne, childOne);
      InsertNodeIntoTree(treeTwo, childTwo);

      child->SoftDelete();
    }
  }

  assert(treeOne->NumChildren() + treeTwo->NumChildren() != 0);

  if (treeOne->NumChildren() == 0)
    AddFakeNodes(treeTwo, treeOne);
  else if (treeTwo->NumChildren() == 0)
    AddFakeNodes(treeOne, treeTwo);

  assert(treeOne->NumChildren() <= treeOne->MaxNumChildren());
  assert(treeTwo->NumChildren() <= treeTwo->MaxNumChildren());
}

template<typename SplitPolicyType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType>::
AddFakeNodes(const TreeType* tree, TreeType* emptyTree)
{
  size_t numDescendantNodes = 1;

  TreeType* node = tree->Children()[0];

  while (!node->IsLeaf())
  {
    numDescendantNodes++;
    node = node->Children()[0];
  }

  node = emptyTree;
  for (size_t i = 0; i < numDescendantNodes; i++)
  {
    TreeType* child = new TreeType(node);

    node = child;
  }
}


template<typename SplitPolicyType>
template<typename TreeType>
bool RPlusTreeSplit<SplitPolicyType>::CheckNonLeafSweep(const TreeType* node,
    size_t cutAxis, double cut)
{
  size_t numTreeOneChildren = 0;
  size_t numTreeTwoChildren = 0;

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    TreeType* child = node->Children()[i];
    int policy = SplitPolicyType::GetSplitPolicy(child, cutAxis, cut);
    if (policy == SplitPolicyType::AssignToFirstTree)
      numTreeOneChildren++;
    else if (policy == SplitPolicyType::AssignToSecondTree)
      numTreeTwoChildren++;
    else
    {
      numTreeOneChildren++;
      numTreeTwoChildren++;
    }
  }

  if (numTreeOneChildren <= node->MaxNumChildren() && numTreeOneChildren > 0 &&
      numTreeTwoChildren <= node->MaxNumChildren() && numTreeTwoChildren > 0)
    return true;
  return false;
}

template<typename SplitPolicyType>
template<typename TreeType>
bool RPlusTreeSplit<SplitPolicyType>::CheckLeafSweep(const TreeType* node,
    size_t cutAxis, double cut)
{
  size_t numTreeOnePoints = 0;
  size_t numTreeTwoPoints = 0;

  for (size_t i = 0; i < node->NumPoints(); i++)
  {
    if (node->Dataset().col(node->Point(i))[cutAxis] <= cut)
      numTreeOnePoints++;
    else
      numTreeTwoPoints++;
  }

  if (numTreeOnePoints <= node->MaxLeafSize() && numTreeOnePoints > 0 &&
      numTreeTwoPoints <= node->MaxLeafSize() && numTreeTwoPoints > 0)
    return true;
  return false;
}

template<typename SplitPolicyType>
template<typename TreeType>
bool RPlusTreeSplit<SplitPolicyType>::PartitionNode(const TreeType* node, size_t fillFactor,
    size_t& minCutAxis, double& minCut)
{
  if ((node->NumChildren() <= fillFactor && !node->IsLeaf()) ||
      (node->Count() <= fillFactor && node->IsLeaf()))
    return false;

  double minCost = std::numeric_limits<double>::max();
  minCutAxis = node->Bound().Dim();

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    double cut;
    double cost;

    if (node->IsLeaf())
      cost = SweepLeafNode(k, node, fillFactor, cut);
    else
      cost = SweepNonLeafNode(k, node, fillFactor, cut);
    

    if (cost < minCost)
    {
      minCost = cost;
      minCutAxis = k;
      minCut = cut;      
    }
  }
  return true;
}

template<typename SplitPolicyType>
template<typename TreeType>
double RPlusTreeSplit<SplitPolicyType>::SweepNonLeafNode(size_t axis, const TreeType* node,
    size_t fillFactor, double& axisCut)
{
  typedef typename TreeType::ElemType ElemType;

  std::vector<SortStruct<ElemType>> sorted(node->NumChildren());

  for (size_t i = 0; i < node->NumChildren(); i++)
  {
    sorted[i].d = SplitPolicyType::Bound(node->Children()[i])[axis].Hi();
    sorted[i].n = i;
  }
  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

  size_t splitPointer = fillFactor;

  axisCut = sorted[splitPointer - 1].d;

  if (!CheckNonLeafSweep(node, axis, axisCut))
  {
    for (splitPointer = 1; splitPointer < node->NumChildren(); splitPointer++)
    {
      axisCut = sorted[splitPointer - 1].d;
      if (CheckNonLeafSweep(node, axis, axisCut))
        break;
    }

    if (splitPointer == node->NumChildren())
      return std::numeric_limits<double>::max();
  }

  std::vector<ElemType> lowerBound1(node->Bound().Dim());
  std::vector<ElemType> highBound1(node->Bound().Dim());
  std::vector<ElemType> lowerBound2(node->Bound().Dim());
  std::vector<ElemType> highBound2(node->Bound().Dim());

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    lowerBound1[k] = node->Children()[sorted[0].n]->Bound()[k].Lo();
    highBound1[k] = node->Children()[sorted[0].n]->Bound()[k].Hi();

    for (size_t i = 1; i < splitPointer; i++)
    {
      if (node->Children()[sorted[i].n]->Bound()[k].Lo() < lowerBound1[k])
        lowerBound1[k] = node->Children()[sorted[i].n]->Bound()[k].Lo();
      if (node->Children()[sorted[i].n]->Bound()[k].Hi() > highBound1[k])
        highBound1[k] = node->Children()[sorted[i].n]->Bound()[k].Hi();
    }

    lowerBound2[k] = node->Children()[sorted[splitPointer].n]->Bound()[k].Lo();
    highBound2[k] = node->Children()[sorted[splitPointer].n]->Bound()[k].Hi();

    for (size_t i = splitPointer + 1; i < node->NumChildren(); i++)
    {
      if (node->Children()[sorted[i].n]->Bound()[k].Lo() < lowerBound2[k])
        lowerBound2[k] = node->Children()[sorted[i].n]->Bound()[k].Lo();
      if (node->Children()[sorted[i].n]->Bound()[k].Hi() > highBound2[k])
        highBound2[k] = node->Children()[sorted[i].n]->Bound()[k].Hi();
    }
  }

  ElemType area1 = 1.0, area2 = 1.0;
  ElemType overlappedArea = 1.0;

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    if (lowerBound1[k] >= highBound1[k])
    {
      overlappedArea *= 0;
      area1 *= 0;
    }
    else
      area1 *= highBound1[k] - lowerBound1[k];

    if (lowerBound2[k] >= highBound2[k])
    {
      overlappedArea *= 0;
      area1 *= 0;
    }
    else
      area2 *= highBound2[k] - lowerBound2[k];

    if (lowerBound1[k] < highBound1[k] && lowerBound2[k] < highBound2[k])
    {
      if (lowerBound1[k] > highBound2[k] || lowerBound2[k] > highBound2[k])
        overlappedArea *= 0;
      else
        overlappedArea *= std::min(highBound1[k], highBound2[k]) -
            std::max(lowerBound1[k], lowerBound2[k]);
    }
  }

  return area1 + area2 - overlappedArea;
}

template<typename SplitPolicyType>
template<typename TreeType>
double RPlusTreeSplit<SplitPolicyType>::SweepLeafNode(size_t axis, const TreeType* node,
    size_t fillFactor, double& axisCut)
{
  typedef typename TreeType::ElemType ElemType;

  std::vector<SortStruct<ElemType>> sorted(node->Count());

  sorted.resize(node->Count());

  for (size_t i = 0; i < node->NumPoints(); i++)
  {
    sorted[i].d = node->Dataset().col(node->Point(i))[axis];
    sorted[i].n = i;
  }

  std::sort(sorted.begin(), sorted.end(), StructComp<ElemType>);

  axisCut = sorted[fillFactor - 1].d;

  if (!CheckLeafSweep(node, axis, axisCut))
    return std::numeric_limits<double>::max();

  std::vector<ElemType> lowerBound1(node->Bound().Dim());
  std::vector<ElemType> highBound1(node->Bound().Dim());
  std::vector<ElemType> lowerBound2(node->Bound().Dim());
  std::vector<ElemType> highBound2(node->Bound().Dim());

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    lowerBound1[k] = node->Dataset().col(node->Point(sorted[0].n))[k];
    highBound1[k] = node->Dataset().col(node->Point(sorted[0].n))[k];

    for (size_t i = 1; i < fillFactor; i++)
    {
      if (node->Dataset().col(node->Point(sorted[i].n))[k] < lowerBound1[k])
        lowerBound1[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
      if (node->Dataset().col(node->Point(sorted[i].n))[k] > highBound1[k])
        highBound1[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
    }

    lowerBound2[k] = node->Dataset().col(node->Point(sorted[fillFactor].n))[k];
    highBound2[k] = node->Dataset().col(node->Point(sorted[fillFactor].n))[k];

    for (size_t i = fillFactor + 1; i < node->NumChildren(); i++)
    {
      if (node->Dataset().col(node->Point(sorted[i].n))[k] < lowerBound2[k])
        lowerBound2[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
      if (node->Dataset().col(node->Point(sorted[i].n))[k] > highBound2[k])
        highBound2[k] = node->Dataset().col(node->Point(sorted[i].n))[k];
    }
  }

  ElemType area1 = 1.0, area2 = 1.0;
  ElemType overlappedArea = 1.0;

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    area1 *= highBound1[k] - lowerBound1[k];
    area2 *= highBound2[k] - lowerBound2[k];
  }

  return area1 + area2 - overlappedArea;
}

template<typename SplitPolicyType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType>::
InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->Children()[destTree->NumChildren()++] = srcNode;
}


} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP
