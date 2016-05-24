/**
 * @file hilbert_r_tree_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Defintion of the HilbertRTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_HILBERT_R_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

template<typename TreeType,typename HilbertValue>
class HilbertRTreeSplit
{
 public:
  //! Default constructor
  HilbertRTreeSplit();

  //! Construct this with the specified node.
  HilbertRTreeSplit(const TreeType *node);

  //! Create a copy of the other.split.
  HilbertRTreeSplit(const TreeType &other);

  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   */
  void SplitLeafNode(TreeType *tree,std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  bool SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels);
 private:
  HilbertValue largestHilbertValue;
  const int splitOrder = 2;

 public:
  HilbertValue &LargestHilbertValue() { return largestHilbertValue };

  HilbertValue LargestHilbertValue() { return largestHilbertValue } const;

  bool FindCooperatingSiblings(TreeType *parent,size_t iTree,size_t &firstSubling,size_t &lastSibling);

  void RedistributeNodesEvenly(const TreeType *parent,size_t firstSibling,size_t lastSibling);

  void RedistributePointsEvenly(const TreeType *parent,size_t firstSibling,size_t lastSibling);


 public:
  /**
   * Serialize the split.
   */
  template<typename Archive>
  void Serialize(Archive &, const unsigned int /* version */);

};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "hilbert_r_tree_split_impl.hpp"

#endif

