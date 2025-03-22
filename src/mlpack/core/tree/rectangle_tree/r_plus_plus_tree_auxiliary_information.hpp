/**
 * @file core/tree/rectangle_tree/r_plus_plus_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the RPlusPlusTreeAuxiliaryInformation class,
 * a class that provides some r++-tree specific information
 * about the nodes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_AUXILIARY_INFO_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_AUXILIARY_INFO_HPP

#include <mlpack/prereqs.hpp>
#include "../hrectbound.hpp"

namespace mlpack {

template<typename TreeType>
class RPlusPlusTreeAuxiliaryInformation
{
 public:
  //! The element type held by the tree.
  using ElemType = typename TreeType::ElemType;
  //! The bound type held by the auxiliary information.
  using BoundType = HRectBound<EuclideanDistance, ElemType>;

  //! Construct the auxiliary information object.
  RPlusPlusTreeAuxiliaryInformation();

  /**
   * Construct this as an auxiliary information for the given node.
   *
   * @param * (node) The node that stores this auxiliary information.
   */
  RPlusPlusTreeAuxiliaryInformation(const TreeType* /* node */);

  /**
   * Create an auxiliary information object by copying from another object.
   *
   * @param other Another auxiliary information object from which the
   *    information will be copied.
   * @param tree The node that holds the auxiliary information.
   * @param * (deepCopy) If false, the new object uses the same memory
   *    (not used here).
   */
  RPlusPlusTreeAuxiliaryInformation(
      const RPlusPlusTreeAuxiliaryInformation& other,
      TreeType* tree,
      bool /* deepCopy */ = true);

  /**
   * Create an auxiliary information object by moving from another node.
   *
   * @param other The auxiliary information object from which the information
   * will be moved.
   */
  RPlusPlusTreeAuxiliaryInformation(RPlusPlusTreeAuxiliaryInformation&& other);

  /**
   * Copy the given RPlusPlusTreeAuxiliaryInformation.
   */
  RPlusPlusTreeAuxiliaryInformation& operator=(
      const RPlusPlusTreeAuxiliaryInformation& other);

  /**
   * Take ownership of the given RPlusPlusTreeAuxiliaryInformation's data.
   */
  RPlusPlusTreeAuxiliaryInformation& operator=(
      RPlusPlusTreeAuxiliaryInformation&& other);

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the insertion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param * (node) The node in which the point is being inserted.
   * @param * (point) The global number of the point being inserted.
   */
  bool HandlePointInsertion(TreeType* /* node */, const size_t /* point */);

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the insertion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param * (node) The node in which the nodeToInsert is being inserted.
   * @param * (nodeToInsert) The node being inserted.
   * @param * (insertionLevel) The level of the tree at which the nodeToInsert
   *        should be inserted.
   */
  bool HandleNodeInsertion(TreeType* /* node */,
                           TreeType* /* nodeToInsert */,
                           bool /* insertionLevel */);

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the deletion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param * (node) The node from which the point is being deleted.
   * @param * (localIndex) The local index of the point being deleted.
   */
  bool HandlePointDeletion(TreeType* /* node */, const size_t /* localIndex */);

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the deletion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param * (node) The node from which the node is being deleted.
   * @param * (nodeIndex) The local index of the node being deleted.
   */
  bool HandleNodeRemoval(TreeType* /* node */, const size_t /* nodeIndex */);


  /**
   * Some tree types require to propagate the information upward.
   * This method should return false if this is not the case. If true is
   * returned, the update will be propagated upward.
   *
   * @param * (node) The node in which the auxiliary information being update.
   */
  bool UpdateAuxiliaryInfo(TreeType* /* node */);

  /**
   * The R++ tree requires to split the maximum bounding rectangle of a node
   * that is being split. This method is intended for that.
   *
   * @param treeOne The first subtree.
   * @param treeTwo The second subtree.
   * @param axis The axis along which the split is performed.
   * @param cut The coordinate at which the node is split.
   */
  void SplitAuxiliaryInfo(TreeType* treeOne,
                          TreeType* treeTwo,
                          const size_t axis,
                          const ElemType cut);

  /**
   * Nullify the auxiliary information in order to prevent an invalid free.
   */
  void NullifyData();

  //! Return the maximum bounding rectangle.
  BoundType& OuterBound() { return outerBound; }

  //! Modify the maximum bounding rectangle.
  const BoundType& OuterBound() const { return outerBound; }

 private:
  //! The maximum bounding rectangle.
  BoundType outerBound;

 public:
  /**
   * Serialize the information.
   */
  template<typename Archive>
  void serialize(Archive &, const uint32_t /* version */);
};

} // namespace mlpack

#include "r_plus_plus_tree_auxiliary_information_impl.hpp"

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_RPP_TREE_AUXILIARY_INFO_HPP
