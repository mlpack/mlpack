/**
 * @file r_plus_plus_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of the RPlusPlusTreeAuxiliaryInformation class,
 * a class that provides some r++-tree specific information
 * about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_IMPL_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_IMPL_HPP

#include "r_plus_plus_tree_auxiliary_information.hpp"

namespace mlpack {
namespace tree {

template<typename  TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation() :
    outerBound(0)
{

}

template<typename  TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation(const TreeType* tree) :
    outerBound(tree->Parent() ?
               tree->Parent()->AuxiliaryInfo().OuterBound() :
               tree->Bound().Dim())
{
  if (!tree->Parent())
    for (size_t k = 0; k < outerBound.Dim(); k++)
    {
      outerBound[k].Lo() = std::numeric_limits<ElemType>::lowest();
      outerBound[k].Hi() = std::numeric_limits<ElemType>::max();
    }
}

template<typename  TreeType>
RPlusPlusTreeAuxiliaryInformation<TreeType>::
RPlusPlusTreeAuxiliaryInformation(const RPlusPlusTreeAuxiliaryInformation& other) :
    outerBound(other.OuterBound())
{

}

template<typename  TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::
HandlePointInsertion(TreeType* , const size_t )
{
  return false;
}

template<typename  TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::
HandleNodeInsertion(TreeType* , TreeType* ,bool)
{
  assert(false);
  return false;
}

template<typename  TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::
HandlePointDeletion(TreeType* , const size_t)
{
  return false;
}

template<typename  TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::
HandleNodeRemoval(TreeType* , const size_t)
{
  return false;
}

template<typename  TreeType>
bool RPlusPlusTreeAuxiliaryInformation<TreeType>::
UpdateAuxiliaryInfo(TreeType* )
{
  return false;
}

template<typename  TreeType>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::
SplitAuxiliaryInfo(TreeType* treeOne, TreeType* treeTwo, size_t axis,
    typename TreeType::ElemType cut)
{
  typedef bound::HRectBound<metric::EuclideanDistance, ElemType> Bound;
  Bound& treeOneBound = treeOne->AuxiliaryInfo().OuterBound();
  Bound& treeTwoBound = treeTwo->AuxiliaryInfo().OuterBound();

  treeOneBound = outerBound;
  treeTwoBound = outerBound;

  treeOneBound[axis].Hi() = cut;
  treeTwoBound[axis].Lo() = cut;
}


template<typename  TreeType>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::
Copy(TreeType* dst, const TreeType* src)
{
  dst.OuterBound() = src.OuterBound();
}

template<typename  TreeType>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::
NullifyData()
{

}

/**
 * Serialize the information.
 */
template<typename  TreeType>
template<typename Archive>
void RPlusPlusTreeAuxiliaryInformation<TreeType>::
Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(outerBound, "outerBound");
}

} // namespace tree
} // namespace mlpack

#endif//MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_IMPL_HPP
