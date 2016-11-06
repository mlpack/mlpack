/**
 * @file no_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the NoAuxiliaryInformation class, a class that provides
 * no additional information about the nodes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP

namespace mlpack {
namespace tree {

template<typename TreeType>
class NoAuxiliaryInformation
{
 public:
  //! Construct the auxiliary information object.
  NoAuxiliaryInformation() { };
  //! Construct the auxiliary information object.
  NoAuxiliaryInformation(const TreeType* /* node */) { };
  //! Construct the auxiliary information object.
  NoAuxiliaryInformation(const NoAuxiliaryInformation& /* other */,
                         TreeType* /* tree */,
                         bool /* deepCopy */ = true) { };
  //! Construct the auxiliary information object.
  NoAuxiliaryInformation(NoAuxiliaryInformation&& /* other */) { };

  //! Copy the auxiliary information object.
  NoAuxiliaryInformation& operator=(const NoAuxiliaryInformation& /* other */)
  {
    return *this;
  }

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the insertion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param node The node in which the point is being inserted.
   * @param point The global number of the point being inserted.
   */
  bool HandlePointInsertion(TreeType* /* node */, const size_t /* point */)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the insertion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the insertion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param node The node in which the nodeToInsert is being inserted.
   * @param nodeToInsert The node being inserted.
   * @param insertionLevel The level of the tree at which the nodeToInsert
   *        should be inserted.
   */
  bool HandleNodeInsertion(TreeType* /* node */,
                           TreeType* /* nodeToInsert */,
                           bool /* insertionLevel */)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the deletion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param node The node from which the point is being deleted.
   * @param localIndex The local index of the point being deleted.
   */
  bool HandlePointDeletion(TreeType* /* node */, const size_t /* localIndex */)
  {
    return false;
  }

  /**
   * Some tree types require to save some properties at the deletion process.
   * This method allows the auxiliary information the option of manipulating
   * the tree in order to perform the deletion process. If the auxiliary
   * information does that, then the method should return true; if the method
   * returns false the RectangleTree performs its default behavior.
   *
   * @param node The node from which the node is being deleted.
   * @param nodeIndex The local index of the node being deleted.
   */
  bool HandleNodeRemoval(TreeType* /* node */, const size_t /* nodeIndex */)
  {
    return false;
  }

  /**
   * Some tree types require to propagate the information upward.
   * This method should return false if this is not the case. If true is
   * returned, the update will be propagated upward.
   *
   * @param node The node in which the auxiliary information being update.
   */
  bool UpdateAuxiliaryInfo(TreeType* /* node */)
  {
    return false;
  }

  /**
   * The R++ tree requires to split the maximum bounding rectangle of a node
   * that is being split. This method is intended for that. This method is only
   * necessary for an AuxiliaryInformationType that is being used in conjunction
   * with RPlusTreeSplit.
   *
   * @param treeOne The first subtree.
   * @param treeTwo The second subtree.
   * @param axis The axis along which the split is performed.
   * @param cut The coordinate at which the node is split.
   */
  void SplitAuxiliaryInfo(TreeType* /* treeOne */,
                          TreeType* /* treeTwo */,
                          size_t /* axis */,
                          typename TreeType::ElemType /* cut */)
  { }


  /**
   * Nullify the auxiliary information in order to prevent an invalid free.
   */
  void NullifyData()
  { }

  /**
   * Serialize the information.
   */
  template<typename Archive>
  void Serialize(Archive &, const unsigned int /* version */) { };
};

} // namespace tree
} // namespace mlpack

#endif  //  MLPACK_CORE_TREE_RECTANGLE_TREE_NO_AUXILIARY_INFORMATION_HPP
