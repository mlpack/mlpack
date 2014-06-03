/**
 * @file r_tree_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the RTreeSplit class, a class that splits the nodes of an R tree, starting
 * at a leaf node and moving upwards if necessary.
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
template<typename MatType = arma::mat>
class RTreeSplit
{
public:

/**
 * Split a leaf node using the "default" algorithm.  If necessary, this split will propagate
 * upwards through the tree.  The methods for splitting non-leaf nodes are private since
 * they should only be called if a leaf node overflows.
 */
static bool SplitLeafNode(const RectangleTree& tree);

private:

/**
 * Split a non-leaf node using the "default" algorithm.  If this is the root node and
 * we need to move up the tree, a new root node is created.
 */
static bool SplitNonLeafNode(const RectangleTree& tree);

/**
 * Get the seeds for splitting a leaf node.
 */
static void GetPointSeeds(const RectangleTree& tree, int &i, int &j);

/**
 * Get the seeds for splitting a non-leaf node.
 */
static void GetBoundSeeds(const RectangleTree& tree, int &i, int &j);

/**
 * Assign points to the two new nodes.
 */
static void AssignPointDestNode(
    const RectangleTree& oldTree,
    RectangleTree& treeOne,
    RectangleTree& treeTwo,
    const int intI,
    const int intJ);

/**
 * Assign nodes to the two new nodes.
 */
static void AssignNodeDestNode(
    const RectangleTree& oldTree,
    RectangleTree& treeOne,
    RectangleTree& treeTwo,
    const int intI,
    const int intJ);

/**
  * Insert a node into another node.
  */
static void insertNodeIntoTree(
    RectangleTree& destTree,
    RectangleTree& srcNode);

}; // namespace tree
}; // namespace mlpack

// Include implementation
#include "r_tree_split_impl.hpp"

#endif


