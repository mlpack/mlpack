/**
 * @file r_plus_plus_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the RPlusPlusTreeAuxiliaryInformation class,
 * a class that provides some r++-tree specific information
 * about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_HPP

#include <mlpack/core.hpp>
#include "../hrectbound.hpp"

namespace mlpack {
namespace tree {

template<typename TreeType>
class RPlusPlusTreeAuxiliaryInformation
{
 public:
  typedef typename TreeType::ElemType ElemType;

  RPlusPlusTreeAuxiliaryInformation();
  RPlusPlusTreeAuxiliaryInformation(const TreeType* );
  RPlusPlusTreeAuxiliaryInformation(const RPlusPlusTreeAuxiliaryInformation& );

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method should return false if it does not handle the process.
   */
  bool HandlePointInsertion(TreeType* , const size_t);

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method should return false if it does not handle the process.
   */
  bool HandleNodeInsertion(TreeType* , TreeType* ,bool);

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method should return false if it does not handle the process.
   */
  bool HandlePointDeletion(TreeType* , const size_t);

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method should return false if it does not handle the process.
   */
  bool HandleNodeRemoval(TreeType* , const size_t);


  /**
   * Some tree types require to propagate the information downward.
   * This method should return false if this is not the case.
   */
  bool UpdateAuxiliaryInfo(TreeType* );

  void SplitAuxiliaryInfo(TreeType* treeOne, TreeType* treeTwo,
      size_t axis, ElemType cut);

  static void Copy(TreeType* ,const TreeType* );

  void NullifyData();

  
  bound::HRectBound<metric::EuclideanDistance, ElemType>& OuterBound()
  { return outerBound; }

  const bound::HRectBound<metric::EuclideanDistance, ElemType>& OuterBound() const
  { return outerBound; }
 private:

  bound::HRectBound<metric::EuclideanDistance, ElemType> outerBound;
 public:
  /**
   * Serialize the information.
   */
  template<typename Archive>
  void Serialize(Archive &, const unsigned int /* version */);
};

} // namespace tree
} // namespace mlpack

#include "r_plus_plus_tree_auxiliary_information_impl.hpp"

#endif//MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_PLUS_TREE_AUXILIARY_INFORMATION_HPP
