/**
 * @file r_tree_star_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the RStarTreeSplit class, a class that splits the nodes of an R tree, starting
 * at a leaf node and moving upwards if necessary.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_STAR_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A Rectangle Tree has new points inserted at the bottom.  When these
 * nodes overflow, we split them, moving up the tree and splitting nodes
 * as necessary.
 */
template<typename DescentType,
         typename StatisticType,
         typename MatType>
class RStarTreeSplit
{
 public:
  typedef RectangleTree<RStarTreeSplit, DescentType, StatisticType, MatType>
      TreeType;

  /**
   * Split a leaf node using the algorithm described in "The R*-tree: An
   * Efficient and Robust Access method for Points and Rectangles."  If
   * necessary, this split will propagate upwards through the tree.
   */
  static void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  static bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);

 private:
  /**
   * Class to allow for faster sorting.
   */
  struct SortStruct
  {
    double d;
    int n;
  };

  /**
   * Comparator for sorting with SortStruct.
   */
  static bool StructComp(const SortStruct& s1, const SortStruct& s2)
  {
    return s1.d < s2.d;
  }

  /**
   * Insert a node into another node.
   */
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
};

}; // namespace tree
}; // namespace mlpack

// Include implementation
#include "r_star_tree_split_impl.hpp"

#endif
