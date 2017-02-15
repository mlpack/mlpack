/**
 * @file r_plus_plus_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the RPlusPlusTreeAuxiliaryInformation class,
 * a class that provides some r++-tree specific information
 * about the nodes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_IMPL_HPP

#include "r_plus_plus_tree_auxiliary_information.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation() :
    outerBound(0)
{

}

template<typename TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation(const TreeType* tree) :
    outerBound(tree->Parent() ?
               tree->Parent()->AuxiliaryInfo().OuterBound() :
               tree->Bound().Dim())
{
  // Initialize the maximum bounding rectangle if the node is the root
  if (!tree->Parent())
    for (size_t k = 0; k < outerBound.Dim(); k++)
    {
      outerBound[k].Lo() = std::numeric_limits<ElemType>::lowest();
      outerBound[k].Hi() = std::numeric_limits<ElemType>::max();
    }
}

template<typename TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation(
    const RPlusPlusTreeAuxiliaryInformation& other,
    TreeType* /* tree */,
    bool /* deepCopy */) :
    outerBound(other.OuterBound())
{

}

template<typename TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation(RPlusPlusTreeAuxiliaryInformation&& other) :
    outerBound(std::move(other.outerBound))
{

}

template<typename TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::HandlePointInsertion(
    TreeType* /* node */, const size_t /* point */)
{
  return false;
}

template<typename TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::HandleNodeInsertion(
    TreeType* /* node */,
    TreeType* /* nodeToInsert */,
    bool /* insertionLevel */)
{
  assert(false);
  return false;
}

template<typename TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::HandlePointDeletion(
    TreeType* /* node */, const size_t /* localIndex */)
{
  return false;
}

template<typename TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::HandleNodeRemoval(
    TreeType* /* node */, const size_t /* nodeIndex */)
{
  return false;
}

template<typename TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::UpdateAuxiliaryInfo(
    TreeType* /* node */)
{
  return false;
}

template<typename TreeType>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::SplitAuxiliaryInfo(
    TreeType* treeOne,
    TreeType* treeTwo,
    const size_t axis,
    const typename TreeType::ElemType cut)
{
  typedef bound::HRectBound<metric::EuclideanDistance, ElemType> Bound;
  Bound& treeOneBound = treeOne->AuxiliaryInfo().OuterBound();
  Bound& treeTwoBound = treeTwo->AuxiliaryInfo().OuterBound();

  // Copy the maximum bounding rectangle.
  treeOneBound = outerBound;
  treeTwoBound = outerBound;

  // Set proper limits.
  treeOneBound[axis].Hi() = cut;
  treeTwoBound[axis].Lo() = cut;
}

template<typename TreeType>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::NullifyData()
{

}

/**
 * Serialize the information.
 */
template<typename TreeType>
template<typename Archive>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(outerBound, "outerBound");
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_IMPL_HPP
