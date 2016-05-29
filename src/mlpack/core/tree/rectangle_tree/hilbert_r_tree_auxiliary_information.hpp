/**
 * @file hilbert_r_tree_auxiliary_information.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the HilbertRTreeAuxiliaryInformation class,
 * a class that provides some Hilbert r-tree specific information
 * about the nodes.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_HPP

namespace mlpack {
namespace tree {

template<typename TreeType,typename HilbertValue>
class HilbertRTreeAuxiliaryInformation
{
 public:
  HilbertRTreeAuxiliaryInformation();

  HilbertRTreeAuxiliaryInformation(const TreeType *node);

  HilbertRTreeAuxiliaryInformation(const TreeType &other);
  
  bool HandlePointInsertion(TreeType *node, const size_t point);
  
  bool HandleNodeInsertion(TreeType *node,
                           TreeType *nodeToInsert,bool insertionLevel);

  bool HandlePointDeletion(TreeType *node,const size_t localIndex);

  bool HandleNodeRemoval(TreeType *node,const size_t nodeIndex);

  bool ShrinkAuxiliaryInfo(TreeType *node);

  void Copy(TreeType *dst,TreeType *src);

 private:
  HilbertValue largestHilbertValue;

 public:
  //! Return the largest Hilbert value of a point covered by the node.
  HilbertValue LargestHilbertValue() const { return largestHilbertValue; }
  //! Modify the largest Hilbert value of a point covered by the node.
  HilbertValue& LargestHilbertValue() { return largestHilbertValue; }

  /**
   * Serialize the information.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

};

} // namespace tree
} // namespace mlpack

#include "hilbert_r_tree_auxiliary_information_impl.hpp"

#endif//MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_AUXILIARY_INFORMATION_HPP
