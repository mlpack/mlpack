/**
 * @file r_tree_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the RTreeSplit class, a class that splits the nodes of an R
 * tree, starting at a leaf node and moving upwards if necessary.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A Rectangle Tree has new points inserted at the bottom.  When these
 * nodes overflow, we split them, moving up the tree and splitting nodes
 * as necessary.
 */
class RTreeSplit
{
 public:
  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   */
  template<typename TreeType>
  static void SplitLeafNode(TreeType *tree,std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  template<typename TreeType>
  static bool SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels);

 private:
  /**
   * Get the seeds for splitting a leaf node.
   */
  template<typename TreeType>
  static void GetPointSeeds(const TreeType *tree,int& i, int& j);

  /**
   * Get the seeds for splitting a non-leaf node.
   */
  template<typename TreeType>
  static void GetBoundSeeds(const TreeType *tree,int& i, int& j);

  /**
   * Assign points to the two new nodes.
   */
  template<typename TreeType>
  static void AssignPointDestNode(TreeType* oldTree,
                                  TreeType* treeOne,
                                  TreeType* treeTwo,
                                  const int intI,
                                  const int intJ);

  /**
   * Assign nodes to the two new nodes.
   */
  template<typename TreeType>
  static void AssignNodeDestNode(TreeType* oldTree,
                                 TreeType* treeOne,
                                 TreeType* treeTwo,
                                 const int intI,
                                 const int intJ);

  /**
   * Insert a node into another node.
   */
  template<typename TreeType>
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "r_tree_split_impl.hpp"

#endif
