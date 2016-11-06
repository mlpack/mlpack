/**
 * @file r_plus_tree_split.hpp
 * @author Mikhail Lozhnikov
 *
 * Definition of the RPlusTreeSplit class, a class that splits the nodes of an
 * R+ (or R++) tree, starting at a leaf node and moving upwards if necessary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * The RPlusTreeSplit class performs the split process of a node on overflow.
 *
 * @tparam SplitPolicyType The class that helps to determine the subtree into
 *     which we should insert a child node.
 * @tparam SweepType The class that finds the partition of a node along a given
 *     axis. The partition algorithm tries to find a partition along each axis,
 *     evaluates each partition and chooses the best one.
 */
template<typename SplitPolicyType,
         template<typename> class SweepType>
class RPlusTreeSplit
{
 public:
  typedef SplitPolicyType SplitPolicy;
  /**
   * Split a leaf node using the "default" algorithm.  If necessary, this split
   * will propagate upwards through the tree.
   * @param node. The node that is being split.
   * @param relevels Not used.
   */
  template<typename TreeType>
  static void SplitLeafNode(TreeType* tree, std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   * @param node. The node that is being split.
   * @param relevels Not used.
   */
  template<typename TreeType>
  static bool SplitNonLeafNode(TreeType* tree, std::vector<bool>& relevels);

 private:
  /**
   * Split a leaf node along an axis.
   *
   * @param tree The node that is being split into two new nodes.
   * @param treeOne The first subtree of two resulting subtrees.
   * @param treeOne The second subtree of two resulting subtrees.
   * @param cutAxis The axis along which the node is being split.
   * @param cut The coordinate at which the node is being split.
   */
  template<typename TreeType>
  static void SplitLeafNodeAlongPartition(
      TreeType* tree,
      TreeType* treeOne,
      TreeType* treeTwo,
      const size_t cutAxis,
      const typename TreeType::ElemType cut);

  /**
   * Split a non-leaf node along an axis. This method propagates the split
   * downward up to a leaf node if necessary.
   *
   * @param tree The node that is being split into two new nodes.
   * @param treeOne The first subtree of two resulting subtrees.
   * @param treeOne The second subtree of two resulting subtrees.
   * @param cutAxis The axis along which the node is being split.
   * @param cut The coordinate at which the node is being split.
   */
  template<typename TreeType>
  static void SplitNonLeafNodeAlongPartition(
      TreeType* tree,
      TreeType* treeOne,
      TreeType* treeTwo,
      const size_t cutAxis,
      const typename TreeType::ElemType cut);

  /**
   * This method is used to make sure that the tree has equivalent maximum depth
   * in every branch. The method should be invoked if one of two resulting
   * subtrees is empty after the split process
   * (i.e. the subtree contains no children).
   * The method convert the empty node into an empty subtree (increase the node
   * in depth).
   *
   * @param tree One of two subtrees that is not empty.
   * @param emptyTree The empty subtree.
   */
  template<typename TreeType>
  static void AddFakeNodes(const TreeType* tree, TreeType* emptyTree);

  /**
   * Partition a node using SweepType. This method invokes
   * SweepType::Sweep(Non)LeafNode() for each dimension and chooses the
   * best one. The method returns false if the node needn't partitioning.
   * Overwise, the method returns true. If the method failed in finding
   * an acceptable partition, the minCutAxis will be equal to the number of
   * dimensions.
   *
   * @param node The node that is being split.
   * @param minCutAxis The axis along which the node will be split.
   * @param minCut The coordinate at which the node will be split.
   */
  template<typename TreeType>
  static bool PartitionNode(const TreeType* node,
                            size_t& minCutAxis,
                            typename TreeType::ElemType& minCut);

  /**
   * Insert a node into another node.
   */
  template<typename TreeType>
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "r_plus_tree_split_impl.hpp"

#endif  // MLPACK_CORE_TREE_RECTANGLE_TREE_R_PLUS_TREE_SPLIT_HPP
