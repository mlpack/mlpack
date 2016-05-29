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

const int splitOrder = 2;

class HilbertRTreeSplit
{
 public:
  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   */
  template<typename TreeType>
  void SplitLeafNode(TreeType *tree,std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  template<typename TreeType>
  bool SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels);

 private:
  template<typename TreeType>
  bool FindCooperatingSiblings(TreeType *parent,size_t iTree,size_t &firstSibling,size_t &lastSibling);

  template<typename TreeType>
  void RedistributeNodesEvenly(const TreeType *parent,size_t firstSibling,size_t lastSibling);

  template<typename TreeType>
  void RedistributePointsEvenly(const TreeType *parent,size_t firstSibling,size_t lastSibling);

};
} // namespace tree
} // namespace mlpack

// Include implementation
#include "hilbert_r_tree_split_impl.hpp"

#endif

