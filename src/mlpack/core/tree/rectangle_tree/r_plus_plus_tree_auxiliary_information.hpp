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
  //! The element type held by the tree.
  typedef typename TreeType::ElemType ElemType;

  //! Construct the auxiliary information object.
  RPlusPlusTreeAuxiliaryInformation();

  /**
   * Construct this as an auxiliary information for the given node.
   *
   * @param node The node that stores this auxiliary information.
   */
  RPlusPlusTreeAuxiliaryInformation(const TreeType* /* node */);

  /**
   * Create an auxiliary information object by copying from another node.
   *
   * @param other The auxiliary information object from which the information
   * will be copied.
   */
  RPlusPlusTreeAuxiliaryInformation(
      const RPlusPlusTreeAuxiliaryInformation& other);

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
  bool HandlePointInsertion(TreeType* /* node */, const size_t /* point */);

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
                           bool /* insertionLevel */);

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
  bool HandlePointDeletion(TreeType* /* node */, const size_t /* localIndex */);

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
  bool HandleNodeRemoval(TreeType* /* node */, const size_t /* nodeIndex */);


  /**
   * Some tree types require to propagate the information upward.
   * This method should return false if this is not the case. If true is
   * returned, the update will be propogated upward.
   *
   * @param node The node in which the auxiliary information being update.
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
  bound::HRectBound<metric::EuclideanDistance, ElemType>& OuterBound()
  { return outerBound; }

  //! Modify the maximum bounding rectangle.
  const bound::HRectBound<metric::EuclideanDistance, ElemType>&
      OuterBound() const
  { return outerBound; }
 private:
  //! The maximum bounding rectangle.
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
