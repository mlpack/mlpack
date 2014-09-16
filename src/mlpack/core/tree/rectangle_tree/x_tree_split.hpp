/**
 * @file x_tre_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the XTreeSplit class, a class that splits the nodes of an X
 * tree, starting at a leaf node and moving upwards if necessary.
 *
 * This is known to have a bug: see #368.
 */
#ifndef __MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_HPP
#define __MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_HPP

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
class XTreeSplit
{
public:

/**
 * The X-tree paper says that a maximum allowable overlap of 20% works well.
 */
const static double MAX_OVERLAP = 0.2;

/**
 * Split a leaf node using the algorithm described in "The R*-tree: An Efficient and Robust Access method
 * for Points and Rectangles."  If necessary, this split will propagate
 * upwards through the tree.
 */
static void SplitLeafNode(RectangleTree<XTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* tree, std::vector<bool>& relevels);

/**
 * Split a non-leaf node using the "default" algorithm.  If this is a root node, the
 * tree increases in depth.
 */
static bool SplitNonLeafNode(RectangleTree<XTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* tree, std::vector<bool>& relevels);

private:
/**
 * Class to allow for faster sorting.
 */
class sortStruct {
public:
  double d;
  int n;
};

/**
 * Comparator for sorting with sortStruct.
 */
static bool structComp(const sortStruct& s1, const sortStruct& s2) {
  return s1.d < s2.d;
}

/**
  * Insert a node into another node.
  */
static void InsertNodeIntoTree(
    RectangleTree<XTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* destTree,
    RectangleTree<XTreeSplit<DescentType, StatisticType, MatType>, DescentType, StatisticType, MatType>* srcNode);

};

}; // namespace tree
}; // namespace mlpack

// Include implementation
#include "x_tree_split_impl.hpp"

#endif
