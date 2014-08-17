/**
 * @file r_tree_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the RTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP

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
class RTreeSplit
{
 public:
  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   */
  static void SplitLeafNode(
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>,
          DescentType, StatisticType, MatType>* tree,
      std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  static bool SplitNonLeafNode(
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>,
          DescentType, StatisticType, MatType>* tree,
      std::vector<bool>& relevels);

 private:
  /**
   * Get the seeds for splitting a leaf node.
   */
  static void GetPointSeeds(
      const RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>& tree,
      int *i,
      int *j);

  /**
   * Get the seeds for splitting a non-leaf node.
   */
  static void GetBoundSeeds(
      const RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>& tree,
      int *i,
      int *j);

  /**
   * Assign points to the two new nodes.
   */
  static void AssignPointDestNode(
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* oldTree,
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* treeOne,
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* treeTwo,
      const int intI,
      const int intJ);

  /**
   * Assign nodes to the two new nodes.
   */
  static void AssignNodeDestNode(
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* oldTree,
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType> *treeOne,
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType> *treeTwo,
      const int intI,
      const int intJ);

  /**
   * Insert a node into another node.
   */
  static void InsertNodeIntoTree(
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* destTree,
      RectangleTree<RTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* srcNode);
};

}; // namespace tree
}; // namespace mlpack

// Include implementation
#include "r_tree_split_impl.hpp"

#endif
