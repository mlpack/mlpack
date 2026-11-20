/**
 * @file core/tree/rectangle_tree/r_star_tree_split.hpp
 * @author Andrew Wells
 *
 * Definition of the RStarTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A Rectangle Tree has new points inserted at the bottom.  When these
 * nodes overflow, we split them, moving up the tree and splitting nodes
 * as necessary.
 */
class RStarTreeSplit
{
 public:
  /**
   * Split a leaf node using the algorithm described in "The R*-tree: An
   * Efficient and Robust Access method for Points and Rectangles."  If
   * necessary, this split will propagate upwards through the tree.
   */
  template <typename TreeType>
  static void SplitLeafNode(TreeType *tree, std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  template <typename TreeType>
  static bool SplitNonLeafNode(TreeType *tree, std::vector<bool>& relevels);

  /**
   * Reinsert any points into the tree, if needed.  This returns the number of
   * points reinserted.
   */
  template<typename TreeType>
  static size_t ReinsertPoints(TreeType* tree, std::vector<bool>& relevels);

  /**
   * Given a node, return the best dimension and the best index to split on.
   */
  template<typename TreeType>
  static void PickLeafSplit(
      TreeType* tree,
      size_t& bestAxis,
      size_t& bestIndex);

 private:
  /**
   * Insert a node into another node.
   */
  template <typename TreeType>
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);

  /**
   * Comparator for sorting with std::pair. This comparator works a little bit
   * faster then the default comparator.
   */
  template<typename ElemType, typename TreeType>
  static bool PairComp(const std::pair<ElemType, TreeType>& p1,
                       const std::pair<ElemType, TreeType>& p2)
  {
    return p1.first < p2.first;
  }
};

} // namespace mlpack

// Include implementation
#include "r_star_tree_split_impl.hpp"

#endif
