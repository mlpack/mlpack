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
  XTreeSplit(const TreeType *node);

  //! Create a copy of the other.split.
  XTreeSplit(const TreeType &other);

  /**
   * Split a leaf node using the algorithm described in "The R*-tree: An
   * Efficient and Robust Access method for Points and Rectangles."  If
   * necessary, this split will propagate upwards through the tree.
   */
  void SplitLeafNode(TreeType *tree,std::vector<bool>& relevels);

  /**
   * Split a non-leaf node using the "default" algorithm.  If this is a root
   * node, the tree increases in depth.
   */
  bool SplitNonLeafNode(TreeType *tree,std::vector<bool>& relevels);

  /**
   * The X tree requires that the tree records it's "split history".  To make
   * this easy, we use the following structure.
   */
  typedef struct SplitHistoryStruct
  {
    int lastDimension;
    std::vector<bool> history;

    SplitHistoryStruct(int dim) : lastDimension(0), history(dim)
    {
      for (int i = 0; i < dim; i++)
        history[i] = false;
    }

    template<typename Archive>
    void Serialize(Archive& ar, const unsigned int /* version */)
    {
      ar & data::CreateNVP(lastDimension, "lastDimension");
      ar & data::CreateNVP(history, "history");
    }
  } SplitHistoryStruct;

 private:
  //! The max number of child nodes a non-leaf normal node can have.
  size_t normalNodeMaxNumChildren;
  //! A struct to store the "split history" for X trees.
  SplitHistoryStruct splitHistory;

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
  //! Return the split history of the node assosiated with this object.
  const SplitHistoryStruct& SplitHistory() const { return splitHistory; }
  //! Modify the split history of the node assosiated with this object.
  SplitHistoryStruct& SplitHistory() { return splitHistory; }


  /**
   * Serialize the split.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);
};

} // namespace tree
} // namespace mlpack

// Include implementation
#include "x_tree_split_impl.hpp"

#endif
