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

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType, SweepType>::
SplitLeafNode(TreeType* tree, std::vector<bool>& relevels)
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

  size_t cutAxis;
  typename TreeType::ElemType cut;

  if ( !PartitionNode(tree, cutAxis, cut))
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

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
bool RPlusTreeSplit<SplitPolicyType, SweepType>::
SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels)
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
  size_t cutAxis;
  typename TreeType::ElemType cut;

  if ( !PartitionNode(tree, cutAxis, cut))
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

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType, SweepType>::
SplitLeafNodeAlongPartition(TreeType* tree, TreeType* treeOne,
    TreeType* treeTwo, size_t cutAxis, typename TreeType::ElemType cut)
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

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType, SweepType>::
SplitNonLeafNodeAlongPartition(TreeType* tree, TreeType* treeOne,
    TreeType* treeTwo, size_t cutAxis, typename TreeType::ElemType cut)
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

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType, SweepType>::
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

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
bool RPlusTreeSplit<SplitPolicyType, SweepType>::
PartitionNode(const TreeType* node, size_t& minCutAxis,
    typename TreeType::ElemType& minCut)
{
  if ((node->NumChildren() <= fillFactor && !node->IsLeaf()) ||
      (node->Count() <= fillFactor && node->IsLeaf()))
    return false;

  typedef typename SweepType<SplitPolicyType>::template SweepCost<TreeType>::type
      SweepCostType;

  SweepCostType minCost = std::numeric_limits<SweepCostType>::max();
  minCutAxis = node->Bound().Dim();

  for (size_t k = 0; k < node->Bound().Dim(); k++)
  {
    typename TreeType::ElemType cut;
    SweepCostType cost;

    if (node->IsLeaf())
      cost = SweepType<SplitPolicyType>::SweepLeafNode(k, node, cut);
    else
      cost = SweepType<SplitPolicyType>::SweepNonLeafNode(k, node, cut);
    

    if (cost < minCost)
    {
      minCost = cost;
      minCutAxis = k;
      minCut = cut;      
    }
  }
  return true;
}

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplit<SplitPolicyType, SweepType>::
InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->Children()[destTree->NumChildren()++] = srcNode;
}


} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_IMPL_HPP
