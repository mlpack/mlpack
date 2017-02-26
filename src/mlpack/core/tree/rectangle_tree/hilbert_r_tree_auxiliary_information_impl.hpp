/**
 * @file hilbert_r_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the HilbertRTreeAuxiliaryInformation class, a class that
 * provides some Hilbert r-tree specific information about the nodes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP

#include "hilbert_r_tree_auxiliary_information.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation()
{ }

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation(const TreeType* node) :
    hilbertValue(node)
{ }

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation(
    const HilbertRTreeAuxiliaryInformation& other,
    TreeType* tree,
    bool deepCopy) :
    hilbertValue(other.HilbertValue(), tree, deepCopy)
{ }

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HilbertRTreeAuxiliaryInformation(HilbertRTreeAuxiliaryInformation&& other) :
    hilbertValue(std::move(other.hilbertValue))
{ }

template<typename TreeType,
         template<typename> class HilbertValueType>
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>&
HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::operator=(
    const HilbertRTreeAuxiliaryInformation& other)
{
  hilbertValue = other.hilbertValue;
  return *this;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandlePointInsertion(TreeType* node, const size_t point)
{
  if (node->IsLeaf())
  {
    // Get the position at which the point should be inserted, and then update
    // the largest Hilbert value of the node.
    size_t pos = hilbertValue.InsertPoint(node, node->Dataset().col(point));

    // Move points.
    for (size_t i = node->NumPoints(); i > pos; i--)
      node->Point(i) = node->Point(i - 1);

    // Insert the point.
    node->Point(pos) = point;
    node->Count()++;
  }
  else
  {
    // Calculate the Hilbert value.
    hilbertValue.InsertPoint(node, node->Dataset().col(point));
  }

  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandleNodeInsertion(TreeType* node, TreeType* nodeToInsert, bool insertionLevel)
{
  if (insertionLevel)
  {
    size_t pos;

    // Find the best position for the node being inserted.
    // The node should be inserted according to its Hilbert value.
    for (pos = 0; pos < node->NumChildren(); pos++)
      if (HilbertValueType<ElemType>::CompareValues(
                 node->Child(pos).AuxiliaryInfo().HilbertValue(),
                 nodeToInsert->AuxiliaryInfo().HilbertValue()) < 0)
          break;

    // Move nodes.
    for (size_t i = node->NumChildren(); i > pos; i--)
      node->children[i] = node->children[i - 1];

    // Insert the node.
    node->children[pos] = nodeToInsert;
    nodeToInsert->Parent() = node;

    // Update the largest Hilbert value.
    hilbertValue.InsertNode(nodeToInsert);
  }
  else
    hilbertValue.InsertNode(nodeToInsert); // Update the largest Hilbert value.

  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandlePointDeletion(TreeType* node, const size_t localIndex)
{
  // Update the largest Hilbert value.
  hilbertValue.DeletePoint(node,localIndex);

  for (size_t i = localIndex + 1; localIndex < node->NumPoints(); i++)
    node->Point(i - 1) = node->Point(i);

  node->NumPoints()--;
  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
HandleNodeRemoval(TreeType* node, const size_t nodeIndex)
{
  // Update the largest Hilbert value.
  hilbertValue.RemoveNode(node,nodeIndex);

  for (size_t i = nodeIndex + 1; nodeIndex < node->NumChildren(); i++)
    node->children[i - 1] = node->children[i];

  node->NumChildren()--;
  return true;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
bool HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
UpdateAuxiliaryInfo(TreeType* node)
{
  if (node->IsLeaf())  //  Should already be updated
    return true;

  TreeType& child = node->Child(node->NumChildren() - 1);
  if (hilbertValue.CompareWith(child.AuxiliaryInfo().HilbertValue()) < 0)
  {
    hilbertValue = child.AuxiliaryInfo().HilbertValue();
    return true;
  }
  return false;
}

template<typename TreeType,
         template<typename> class HilbertValueType>
void HilbertRTreeAuxiliaryInformation<TreeType, HilbertValueType>::
NullifyData()
{
  hilbertValue.NullifyData();
}

template<typename TreeType,
         template<typename> class HilbertValueType>
template<typename Archive>
void HilbertRTreeAuxiliaryInformation<TreeType ,HilbertValueType>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(hilbertValue, "hilbertValue");
}


} // namespace tree
} // namespace mlpack

#endif//MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_IMPL_HPP
