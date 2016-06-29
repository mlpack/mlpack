/**
 * @file no_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the NoAuxiliaryInformation class, a class that provides
 * no additional information about the nodes.
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
  NoAuxiliaryInformation(const TreeType& /* node */) { };

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
  bool HandlePointInsertion(TreeType* , const size_t)
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
   * returned, the update will be propogated upward.
   *
   * @param node The node in which the auxiliary information being update.
   */
  bool UpdateAuxiliaryInfo(TreeType* /* node */)
  {
    return false;
  }

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
