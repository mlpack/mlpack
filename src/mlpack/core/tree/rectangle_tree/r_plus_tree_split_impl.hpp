/**
 * @file core/tree/rectangle_tree/r_plus_tree_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (RPlusTreeSplit) to split a RectangleTree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_IMPL_HPP

#include "r_plus_tree_split.hpp"
#include "r_plus_plus_tree_auxiliary_information.hpp"
#include "r_plus_tree_split_policy.hpp"
#include "r_plus_plus_tree_split_policy.hpp"

namespace mlpack {

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplitType<SplitPolicyType, SweepType>::
SplitLeafNode(TreeType* tree, std::vector<bool>& relevels)
{
  using ElemType = typename TreeType::ElemType;

  if (tree->Count() == 1)
  {
    // Check if an intermediate node was added during the insertion process.
    // i.e. we couldn't enlarge a node of the R+ tree. So, one of intermediate
    // nodes may be overflowed.
    TreeType* node = tree->Parent();

    while (node != NULL)
    {
      if (node->NumChildren() == node->MaxNumChildren() + 1)
      {
        // Split the overflowed node.
        RPlusTreeSplitType::SplitNonLeafNode(node, relevels);
        return;
      }
      node = node->Parent();
    }
    return;
  }
  else if (tree->Count() <= tree->MaxLeafSize())
  {
    return;
  }

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

    RPlusTreeSplitType::SplitLeafNode(copy, relevels);
    return;
  }

  size_t cutAxis = tree->Bound().Dim();
  ElemType cut = std::numeric_limits<ElemType>::lowest();

  // Try to find a partition of the node.
  if (!PartitionNode(tree, cutAxis, cut))
    return;

  // If we could not find a suitable partition.
  if (cutAxis == tree->Bound().Dim())
  {
    tree->MaxLeafSize()++;
    tree->points.resize(tree->MaxLeafSize() + 1);
    Log::Warn << "Could not find an acceptable partition."
        "The size of the node will be increased.";
    return;
  }

  TreeType* treeOne = new TreeType(tree->Parent(), tree->MaxNumChildren());
  TreeType* treeTwo = new TreeType(tree->Parent(), tree->MaxNumChildren());
  treeOne->MinLeafSize() = 0;
  treeOne->MinNumChildren() = 0;
  treeTwo->MinLeafSize() = 0;
  treeTwo->MinNumChildren() = 0;

  // Split the node into two new nodes.
  SplitLeafNodeAlongPartition(tree, treeOne, treeTwo, cutAxis, cut);

  TreeType* parent = tree->Parent();
  size_t i = 0;
  while (parent->children[i] != tree)
    ++i;

  assert(i < parent->NumChildren());

  // Insert two new nodes to the tree.
  parent->children[i] = treeOne;
  parent->children[parent->NumChildren()++] = treeTwo;

  assert(parent->NumChildren() <= parent->MaxNumChildren() + 1);

  // Propagate the split upward if necessary.
  if (parent->NumChildren() == parent->MaxNumChildren() + 1)
    RPlusTreeSplitType::SplitNonLeafNode(parent, relevels);

  tree->SoftDelete();
}

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
bool RPlusTreeSplitType<SplitPolicyType, SweepType>::
SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels)
{
  using ElemType = typename TreeType::ElemType;
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

    RPlusTreeSplitType::SplitNonLeafNode(copy, relevels);
    return true;
  }
  size_t cutAxis = tree->Bound().Dim();
  ElemType cut = std::numeric_limits<ElemType>::lowest();

  // Try to find a partition of the node.
  if ( !PartitionNode(tree, cutAxis, cut))
    return false;

  // If we could not find a suitable partition.
  if (cutAxis == tree->Bound().Dim())
  {
    tree->MaxNumChildren()++;
    tree->children.resize(tree->MaxNumChildren() + 1);
    Log::Warn << "Could not find an acceptable partition."
        "The size of the node will be increased.";
    return false;
  }

  TreeType* treeOne = new TreeType(tree->Parent(), tree->MaxNumChildren());
  TreeType* treeTwo = new TreeType(tree->Parent(), tree->MaxNumChildren());
  treeOne->MinLeafSize() = 0;
  treeOne->MinNumChildren() = 0;
  treeTwo->MinLeafSize() = 0;
  treeTwo->MinNumChildren() = 0;

  // Split the node into two new nodes.
  SplitNonLeafNodeAlongPartition(tree, treeOne, treeTwo, cutAxis, cut);

  TreeType* parent = tree->Parent();
  size_t i = 0;
  while (parent->children[i] != tree)
    ++i;

  assert(i < parent->NumChildren());

  // Insert two new nodes to the tree.
  parent->children[i] = treeOne;
  parent->children[parent->NumChildren()++] = treeTwo;

  tree->SoftDelete();

  assert(parent->NumChildren() <= parent->MaxNumChildren() + 1);

  // Propagate the split upward if necessary.
  if (parent->NumChildren() == parent->MaxNumChildren() + 1)
    RPlusTreeSplitType::SplitNonLeafNode(parent, relevels);

  return false;
}

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplitType<SplitPolicyType, SweepType>::
SplitLeafNodeAlongPartition(
    TreeType* tree,
    TreeType* treeOne,
    TreeType* treeTwo,
    const size_t cutAxis,
    const typename TreeType::ElemType cut)
{
  // Split the auxiliary information.
  tree->AuxiliaryInfo().SplitAuxiliaryInfo(treeOne, treeTwo, cutAxis, cut);

  // Ensure that the capacity of the nodes is sufficient.
  if (treeOne->MaxLeafSize() < tree->NumPoints())
  {
    treeOne->MaxLeafSize() = tree->NumPoints();
    treeOne->points.resize(treeOne->MaxLeafSize() + 1);
  }

  if (treeTwo->MaxLeafSize() < tree->NumPoints())
  {
    treeTwo->MaxLeafSize() = tree->NumPoints();
    treeTwo->points.resize(treeTwo->MaxLeafSize() + 1);
  }

  // Insert points into the corresponding subtree.
  for (size_t i = 0; i < tree->NumPoints(); ++i)
  {
    if (tree->Dataset().col(tree->Point(i))[cutAxis] <= cut)
    {
      treeOne->Point(treeOne->Count()++) = tree->Point(i);
      treeOne->Bound() |= tree->Dataset().col(tree->Point(i));
    }
    else
    {
      treeTwo->Point(treeTwo->Count()++) = tree->Point(i);
      treeTwo->Bound() |= tree->Dataset().col(tree->Point(i));
    }
  }
  // Update the number of descandants.
  treeOne->numDescendants = treeOne->Count();
  treeTwo->numDescendants = treeTwo->Count();

  assert(treeOne->Count() <= treeOne->MaxLeafSize());
  assert(treeTwo->Count() <= treeTwo->MaxLeafSize());

  assert(tree->Count() == treeOne->Count() + treeTwo->Count());
  assert(treeOne->Bound()[cutAxis].Hi() < treeTwo->Bound()[cutAxis].Lo());
}

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
void RPlusTreeSplitType<SplitPolicyType, SweepType>::
SplitNonLeafNodeAlongPartition(
    TreeType* tree,
    TreeType* treeOne,
    TreeType* treeTwo,
    const size_t cutAxis,
    const typename TreeType::ElemType cut)
{
  // Split the auxiliary information.
  tree->AuxiliaryInfo().SplitAuxiliaryInfo(treeOne, treeTwo, cutAxis, cut);

  // Insert children into the corresponding subtree.
  for (size_t i = 0; i < tree->NumChildren(); ++i)
  {
    TreeType* child = tree->children[i];
    int policy = SplitPolicyType::GetSplitPolicy(*child, cutAxis, cut);

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
      // The child should be split (i.e. the partition divides its bound).
      TreeType* childOne = new TreeType(treeOne);
      TreeType* childTwo = new TreeType(treeTwo);
      treeOne->MinLeafSize() = 0;
      treeOne->MinNumChildren() = 0;
      treeTwo->MinLeafSize() = 0;
      treeTwo->MinNumChildren() = 0;

      // Propagate the split downward.
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

  // Add a fake subtree if one of the subtrees is empty.
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
void RPlusTreeSplitType<SplitPolicyType, SweepType>::
AddFakeNodes(const TreeType* tree, TreeType* emptyTree)
{
  size_t numDescendantNodes = tree->TreeDepth() - 1;

  TreeType* node = emptyTree;
  for (size_t i = 0; i < numDescendantNodes; ++i)
  {
    TreeType* child = new TreeType(node);
    node->children[node->NumChildren()++] = child;

    node = child;
  }
}

template<typename SplitPolicyType,
         template<typename> class SweepType>
template<typename TreeType>
bool RPlusTreeSplitType<SplitPolicyType, SweepType>::
PartitionNode(const TreeType* node, size_t& minCutAxis,
    typename TreeType::ElemType& minCut)
{
  if ((node->NumChildren() <= node->MaxNumChildren() && !node->IsLeaf()) ||
      (node->Count() <= node->MaxLeafSize() && node->IsLeaf()))
    return false; // No partition required.

  // Define the type of the sweep cost.
  using SweepCostType = typename
      SweepType<SplitPolicyType>::template SweepCost<TreeType>::type;

  SweepCostType minCost = std::numeric_limits<SweepCostType>::max();
  minCutAxis = node->Bound().Dim();

  // Find the sweep with a minimal cost.
  for (size_t k = 0; k < node->Bound().Dim(); ++k)
  {
    typename TreeType::ElemType cut = 0.0;
    SweepCostType cost = 0.0;

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
void RPlusTreeSplitType<SplitPolicyType, SweepType>::
InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode)
{
  destTree->Bound() |= srcNode->Bound();
  destTree->numDescendants += srcNode->numDescendants;
  destTree->children[destTree->NumChildren()++] = srcNode;
}

} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_IMPL_HPP
