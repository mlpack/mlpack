/**
 * @file r_tree_star_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the RStarTreeSplit class, a class that splits the nodes of an R tree, starting
 * at a leaf node and moving upwards if necessary.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A Rectangle Tree has new points inserted at the bottom.  When these
 * nodes overflow, we split them, moving up the tree and splitting nodes
 * as necessary.
 */
template <typename TreeType>
class RStarTreeSplit
{
 public:
  //! Default constructor.
  RStarTreeSplit() { }

  //! Construct this with the specified node.
  RStarTreeSplit(const TreeType* /* node */) { }

  //! Create a copy of the other.split.
  RStarTreeSplit(const TreeType& /* other */) { }

  /**
   * Split a leaf node using the algorithm described in "The R*-tree: An
   * Efficient and Robust Access method for Points and Rectangles."  If
   * necessary, this split will propagate upwards through the tree.
   */
  void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);

 private:
  /**
   * Insert a node into another node.
   */
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);

 public:
  /**
   * Serialize the split.
   */
  template<typename Archive>
  void Serialize(Archive &, const unsigned int /* version */) { };

  /**
   * Comparator for sorting with std::pair. This comparator works a little bit
   * faster then the default comparator.
   */
  template<typename ElemType>
  static bool PairComp(const std::pair<ElemType, size_t>& p1,
                       const std::pair<ElemType, size_t>& p2)
  {
    return p1.first < p2.first;
  }
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "r_star_tree_split_impl.hpp"

#endif
