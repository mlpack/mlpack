/**
 * @file core/tree/rectangle_tree/hilbert_r_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the HilbertRTreeAuxiliaryInformation class,
 * a class that provides some Hilbert r-tree specific information
 * about the nodes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_AUXILIARY_INFO_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_AUXILIARY_INFO_HPP

namespace mlpack {

template<typename TreeType,
         template<typename> class HilbertValueType>
class HilbertRTreeAuxiliaryInformation
{
 public:
  //! The element type held by the tree.
  using ElemType = typename TreeType::ElemType;
  //! Default constructor
  HilbertRTreeAuxiliaryInformation();

  /**
   * Construct this as an auxiliary information for the given node.
   *
   * @param node The node that stores this auxiliary information.
   */
  HilbertRTreeAuxiliaryInformation(const TreeType* node);

  /**
   * Create an auxiliary information object by copying from another object.
   *
   * @param other Another auxiliary information object from which the
   *    information will be copied.
   * @param tree The node that holds the auxiliary information.
   * @param deepCopy If false, the new object uses the same memory
   *    (not used here).
   */
  HilbertRTreeAuxiliaryInformation(
      const HilbertRTreeAuxiliaryInformation& other,
      TreeType* tree = NULL,
      bool deepCopy = true);

  /**
   * Create an auxiliary information object by moving from the other node.
   *
   * @param other The object from which the information will be moved.
   */
  HilbertRTreeAuxiliaryInformation(HilbertRTreeAuxiliaryInformation&& other);

  /**
   * Copy the auxiliary information.
   *
   * @param other The object from which the information will be moved.
   */
  HilbertRTreeAuxiliaryInformation& operator=(
      const HilbertRTreeAuxiliaryInformation& other);

  /**
   * The Hilbert R tree requires to insert points according to their Hilbert
   * value. This method should take care of it.  It returns false if it does
   * nothing and true if it handles the insertion process.
   *
   * @param node The node in which the point is being inserted.
   * @param point The number of the point being inserted.
   */
  bool HandlePointInsertion(TreeType* node, const size_t point);

  /**
   * The Hilbert R tree requires to insert nodes according to their Hilbert
   * value. This method should take care of it.  It returns false if it does
   * nothing and true if it handles the insertion process.
   *
   * @param node The node in which the nodeToInsert is being inserted.
   * @param nodeToInsert The node being inserted.
   * @param insertionLevel The level of the tree at which the nodeToInsert
   *        should be inserted.
   */
  bool HandleNodeInsertion(TreeType* node,
                           TreeType* nodeToInsert,
                           bool insertionLevel);

  /**
   * The Hilbert R tree requires all points to be arranged according to their
   * Hilbert value. This method should take care of saving this property after
   * the deletion process.  It returns false if it does nothing and true if it
   * handles the deletion process.
   *
   * @param node The node from which the point is being deleted.
   * @param localIndex The index of the point being deleted.
   */
  bool HandlePointDeletion(TreeType* node, const size_t localIndex);

  /**
   * The Hilbert R tree requires all nodes to be arranged according to their
   * Hilbert value. This method should take care of saving this property after
   * the deletion process.  It returns false if it does nothing and true if it
   * handles the deletion process.
   *
   * @param node The node from which the node is being deleted.
   * @param nodeIndex The index of the node being deleted.
   */
  bool HandleNodeRemoval(TreeType* node, const size_t nodeIndex);

  /**
   * Update the auxiliary information in the node. The method returns true if
   * the update should be propagated downward.
   *
   * @param node The node in which the auxiliary information being update.
   */
  bool UpdateAuxiliaryInfo(TreeType* node);

  //! Clear memory.
  void NullifyData();

  //! Return the children vector of the tree.
  static const std::vector<TreeType*> Children(const TreeType* tree)
  { return tree->children; }

 private:
  //! The largest Hilbert value of a point enclosed by the node.
  HilbertValueType<ElemType> hilbertValue;

 public:
  //! Return the largest Hilbert value of a point covered by the node.
  const HilbertValueType<ElemType>& HilbertValue() const
  { return hilbertValue; }
  //! Modify the largest Hilbert value of a point covered by the node.
  HilbertValueType<ElemType>& HilbertValue() { return hilbertValue; }

  /**
   * Serialize the information.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
};

} // namespace mlpack

#include "hilbert_r_tree_auxiliary_information_impl.hpp"

#endif // MLPACK_CORE_TREE_RECTANGLE_TREE_HR_TREE_AUXILIARY_INFO_HPP
