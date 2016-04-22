/**
 * @file x_tre_split.hpp
 * @author Andrew Wells
 *
 * Defintion of the XTreeSplit class, a class that splits the nodes of an X
 * tree, starting at a leaf node and moving upwards if necessary.
 *
 * This is known to have a bug: see #368.
 */
#ifndef MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_HPP
#define MLPACK_CORE_TREE_RECTANGLE_TREE_X_TREE_SPLIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * The X-tree paper says that a maximum allowable overlap of 20% works well.
 *
 * This code should eventually be refactored so as to avoid polluting
 * mlpack::tree with this random double.
 */
const double MAX_OVERLAP = 0.2;

/**
 * A Rectangle Tree has new points inserted at the bottom.  When these
 * nodes overflow, we split them, moving up the tree and splitting nodes
 * as necessary.
 */
template<typename TreeType>
class XTreeSplit
{
 public:
  //! Default constructor
  XTreeSplit();

  //! Construct this with the specified node.
  XTreeSplit(TreeType *node);

  //! Construct this with the specified node and the specified normalNodeMaxNumChildren.
  XTreeSplit(TreeType *node,const size_t normalNodeMaxNumChildren);

  //! Construct this with the specified node and the parent of the node.
  XTreeSplit(TreeType *node,const TreeType *parentNode);

  //! Create a copy of the other.split.
  XTreeSplit(TreeType *node,const TreeType &other);

  /**
   * Split a leaf node using the algorithm described in "The R*-tree: An
   * Efficient and Robust Access method for Points and Rectangles."  If
   * necessary, this split will propagate upwards through the tree.
   */
  void SplitLeafNode(std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  bool SplitNonLeafNode(std::vector<bool>& relevels);

 private:
  //! The node which has to be split.
  TreeType *tree;

  //! The max number of child nodes a non-leaf normal node can have.
  size_t normalNodeMaxNumChildren;

  /**
   * Class to allow for faster sorting.
   */
  template<typename ElemType>
  class sortStruct
  {
   public:
    ElemType d;
    int n;
  };

  /**
   * Comparator for sorting with sortStruct.
   */
  template<typename ElemType>
  static bool structComp(const sortStruct<ElemType>& s1, 
                         const sortStruct<ElemType>& s2)
  {
    return s1.d < s2.d;
  }

  /**
   * Insert a node into another node.
   */
  static void InsertNodeIntoTree(TreeType* destTree, TreeType* srcNode);

 public:
  //! Return the maximum number of a normal node's children.
  size_t NormalNodeMaxNumChildren() const { return normalNodeMaxNumChildren; }
  //! Modify the maximum number of a normal node's children.
  size_t& NormalNodeMaxNumChildren() { return normalNodeMaxNumChildren; }
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "x_tree_split_impl.hpp"

#endif
