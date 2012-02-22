/**
 * @file binary_space_tree.hpp
 *
 * Definition of generalized binary space partitioning tree (BinarySpaceTree).
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_HPP

#include <mlpack/core.hpp>

#include "statistic.hpp"

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

/**
 * A binary space partitioning tree, such as a KD-tree or a ball tree.  Once the
 * bound and type of dataset is defined, the tree will construct itself.  Call
 * the constructor with the dataset to build the tree on, and the entire tree
 * will be built.
 *
 * This particular tree does not allow growth, so you cannot add or delete nodes
 * from it.  If you need to add or delete a node, the better procedure is to
 * rebuild the tree entirely.
 *
 * This tree does take one parameter, which is the leaf size to be used.  You
 * can set this at runtime with --tree/leaf_size [leaf_size].  You can also set
 * it in your program using CLI:
 *
 * @code
 * CLI::GetParam<int>("tree/leaf_size") = target_leaf_size;
 * @endcode
 *
 * @param leaf_size Maximum number of points allowed in each leaf.
 *
 * @tparam BoundType The bound used for each node.  The valid types of bounds
 *     and the necessary skeleton interface for this class can be found in
 *     bounds/.
 * @tparam StatisticType Extra data contained in the node.  See statistic.hpp
 *     for the necessary skeleton interface.
 */
template<typename BoundType,
         typename StatisticType = EmptyStatistic,
         typename MatType = arma::mat>
class BinarySpaceTree
{
 private:
  //! The left child node.
  BinarySpaceTree* left;
  //! The right child node.
  BinarySpaceTree* right;
  //! The index of the first point in the dataset contained in this node (and
  //! its children).
  size_t begin;
  //! The number of points of the dataset contained in this node (and its
  //! children).
  size_t count;
  //! The bound object for this node.
  BoundType bound;
  //! Any extra data contained in the node.
  StatisticType stat;
  //! The leaf size.
  size_t leafSize;
  //! The dimension this node split on if it is a parent.
  size_t splitDimension;

 public:
  //! So other classes can use TreeType::Mat.
  typedef MatType Mat;

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will modify the ordering of the points in the dataset!
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param leafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType& data, const size_t leafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will modify the ordering of points in the dataset!  A
   * mapping of the old point indices to the new point indices is filled.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param leafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType& data,
                  std::vector<size_t>& oldFromNew,
                  const size_t leafSize = 20);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will modify the ordering of points in the dataset!  A
   * mapping of the old point indices to the new point indices is filled, as
   * well as a mapping of the new point indices to the old point indices.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *     each old point.
   * @param leafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType& data,
                  std::vector<size_t>& oldFromNew,
                  std::vector<size_t>& newFromOld,
                  const size_t leafSize = 20);

  /**
   * Construct this node on a subset of the given matrix, starting at column
   * begin and using count points.  The ordering of that subset of points
   * will be modified!  This is used for recursive tree-building by the other
   * constructors which don't specify point indices.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param leafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType& data,
                  const size_t begin,
                  const size_t count,
                  const size_t leafSize = 20);

  /**
   * Construct this node on a subset of the given matrix, starting at column
   * begin_in and using count_in points.  The ordering of that subset of points
   * will be modified!  This is used for recursive tree-building by the other
   * constructors which don't specify point indices.
   *
   * A mapping of the old point indices to the new point indices is filled, but
   * it is expected that the vector is already allocated with size greater than
   * or equal to (begin_in + count_in), and if that is not true, invalid memory
   * reads (and writes) will occur.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param leafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType& data,
                  const size_t begin,
                  const size_t count,
                  std::vector<size_t>& oldFromNew,
                  const size_t leafSize = 20);

  /**
   * Construct this node on a subset of the given matrix, starting at column
   * begin_in and using count_in points.  The ordering of that subset of points
   * will be modified!  This is used for recursive tree-building by the other
   * constructors which don't specify point indices.
   *
   * A mapping of the old point indices to the new point indices is filled, as
   * well as a mapping of the new point indices to the old point indices.  It is
   * expected that the vector is already allocated with size greater than or
   * equal to (begin_in + count_in), and if that is not true, invalid memory
   * reads (and writes) will occur.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param begin Index of point to start tree construction with.
   * @param count Number of points to use to construct tree.
   * @param oldFromNew Vector which will be filled with the old positions for
   *     each new point.
   * @param newFromOld Vector which will be filled with the new positions for
   *     each old point.
   * @param leafSize Size of each leaf in the tree.
   */
  BinarySpaceTree(MatType& data,
                  const size_t begin,
                  const size_t count,
                  std::vector<size_t>& oldFromNew,
                  std::vector<size_t>& newFromOld,
                  const size_t leafSize = 20);

  /**
   * Create an empty tree node.
   */
  BinarySpaceTree();

  /**
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any pointers or references
   * to any nodes which are children of this one.
   */
  ~BinarySpaceTree();

  /**
   * Find a node in this tree by its begin and count (const).
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin The begin() of the node to find.
   * @param count The count() of the node to find.
   * @return The found node, or NULL if not found.
   */
  const BinarySpaceTree* FindByBeginCount(size_t begin,
                                          size_t count) const;

  /**
   * Find a node in this tree by its begin and count.
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin The begin() of the node to find.
   * @param count The count() of the node to find.
   * @return The found node, or NULL if not found.
   */
  BinarySpaceTree* FindByBeginCount(size_t begin, size_t count);

  //! Return the bound object for this node.
  const BoundType& Bound() const;
  //! Return the bound object for this node.
  BoundType& Bound();

  //! Return the statistic object for this node.
  const StatisticType& Stat() const;
  //! Return the statistic object for this node.
  StatisticType& Stat();

  //! Return whether or not this node is a leaf (true if it has no children).
  bool IsLeaf() const;

  //! Return the leaf size.
  size_t LeafSize() const;

  //! Fills the tree to the specified level.
  size_t ExtendTree(const size_t level);

  /**
   * Gets the left child of this node.
   */
  BinarySpaceTree* Left() const;

  /**
   * Gets the right child of this node.
   */
  BinarySpaceTree* Right() const;

  /**
  * Returns the dimension this parent's children are split on.
  */
  size_t GetSplitDimension() const;

  /**
   * Obtains the number of nodes in the tree, starting with this.
   */
  size_t TreeSize() const;

  /**
   * Obtains the number of levels below this node in the tree, starting with
   * this.
   */
  size_t TreeDepth() const;

  /**
   * Gets the index of the beginning point of this subset.
   */
  size_t Begin() const;

  /**
   * Gets the index one beyond the last index in the subset.
   */
  size_t End() const;

  /**
   * Gets the number of points in this subset.
   */
  size_t Count() const;

 private:
  /**
   * Private copy constructor, available only to fill (pad) the tree to a
   * specified level.
   */
  BinarySpaceTree(const size_t begin,
                  const size_t count,
                  BoundType bound,
                  StatisticType stat,
                  const int leafSize = 20) :
      left(NULL),
      right(NULL),
      begin(begin),
      count(count),
      bound(bound),
      stat(stat),
      leafSize(leafSize) { }

  BinarySpaceTree* CopyMe()
  {
    return new BinarySpaceTree(begin, count, bound, stat, leafSize);
  }

  /**
   * Splits the current node, assigning its left and right children recursively.
   *
   * @param data Dataset which we are using.
   */
  void SplitNode(MatType& data);

  /**
   * Splits the current node, assigning its left and right children recursively.
   * Also returns a list of the changed indices.
   *
   * @param data Dataset which we are using.
   * @param oldFromNew Vector holding permuted indices.
   */
  void SplitNode(MatType& data, std::vector<size_t>& oldFromNew);

  /**
   * Find the index to split on for this node, given that we are splitting in
   * the given split dimension on the specified split value.
   *
   * @param data Dataset which we are using.
   * @param splitDim Dimension of dataset to split on.
   * @param splitVal Value to split on, in the given split dimension.
   */
  size_t GetSplitIndex(MatType& data, int splitDim, double splitVal);

  /**
   * Find the index to split on for this node, given that we are splitting in
   * the given split dimension on the specified split value.  Also returns a
   * list of the changed indices.
   *
   * @param data Dataset which we are using.
   * @param splitDim Dimension of dataset to split on.
   * @param splitVal Value to split on, in the given split dimension.
   * @param oldFromNew Vector holding permuted indices.
   */
  size_t GetSplitIndex(MatType& data, int splitDim, double splitVal,
      std::vector<size_t>& oldFromNew);
};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "binary_space_tree_impl.hpp"

#endif
