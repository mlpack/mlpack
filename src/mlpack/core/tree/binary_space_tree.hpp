/**
 * @file spacetree.h
 *
 * Definition of generalized binary space partitioning tree (BinarySpaceTree).
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_HPP

#include "statistic.hpp"

#include <armadillo>

namespace mlpack {
namespace tree /** Trees and tree-building procedures. */ {

PARAM_MODULE("tree", "Parameters for the binary space partitioning tree.");
PARAM_INT("leaf_size", "Leaf size used during tree construction.", "tree", 20);

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
 * @tparam TBound The bound used for each node.  The valid types of bounds and
 *     the necessary skeleton interface for this class can be found in bounds/.
 * @tparam TDataset The type of dataset (forced to be arma::mat for now).
 * @tparam TStatistic Extra data contained in the node.  See statistic.h for
 *     the necessary skeleton interface.
 */
template<typename Bound,
         typename Statistic = EmptyStatistic>
class BinarySpaceTree {
 private:
  //! The left child node.
  BinarySpaceTree *left_;
  //! The right child node.
  BinarySpaceTree *right_;
  //! The index of the first point in the dataset contained in this node (and
  //! its children).
  size_t begin_;
  //! The number of points of the dataset contained in this node (and its
  //! children).
  size_t count_;
  //! The bound object for this node.
  Bound bound_;
  //! Any extra data contained in the node.
  Statistic stat_;

 public:
  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will modify the ordering of the points in the dataset!
   *
   * @param data Dataset to create tree from.  This will be modified!
   */
  BinarySpaceTree(arma::mat& data);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will modify the ordering of points in the dataset!  A
   * mapping of the old point indices to the new point indices is filled.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param old_from_new Vector which will be filled with the old positions for
   *     each new point.
   * @param new_from_old Vector which will be filled with the new positions for
   *     each old point.
   */
  BinarySpaceTree(arma::mat& data, std::vector<size_t>& old_from_new);

  /**
   * Construct this as the root node of a binary space tree using the given
   * dataset.  This will modify the ordering of points in the dataset!  A
   * mapping of the old point indices to the new point indices is filled, as
   * well as a mapping of the new point indices to the old point indices.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param old_from_new Vector which will be filled with the old positions for
   *     each new point.
   * @param new_from_old Vector which will be filled with the new positions for
   *     each old point.
   */
  BinarySpaceTree(arma::mat& data,
                  std::vector<size_t>& old_from_new,
                  std::vector<size_t>& new_from_old);

  /**
   * Construct this node on a subset of the given matrix, starting at column
   * begin_in and using count_in points.  The ordering of that subset of points
   * will be modified!  This is used for recursive tree-building by the other
   * constructors which don't specify point indices.
   *
   * @param data Dataset to create tree from.  This will be modified!
   * @param begin_in Index of point to start tree construction with.
   * @param count_in Number of points to use to construct tree.
   */
  BinarySpaceTree(arma::mat& data,
                  size_t begin_in,
                  size_t count_in);

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
   * @param begin_in Index of point to start tree construction with.
   * @param count_in Number of points to use to construct tree.
   * @param old_from_new Vector which will be filled with the old positions for
   *     each new point.
   */
  BinarySpaceTree(arma::mat& data,
                  size_t begin_in,
                  size_t count_in,
                  std::vector<size_t>& old_from_new);

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
   * @param begin_in Index of point to start tree construction with.
   * @param count_in Number of points to use to construct tree.
   * @param old_from_new Vector which will be filled with the old positions for
   *     each new point.
   * @param new_from_old Vector which will be filled with the new positions for
   *     each old point.
   */
  BinarySpaceTree(arma::mat& data,
                  size_t begin_in,
                  size_t count_in,
                  std::vector<size_t>& old_from_new,
                  std::vector<size_t>& new_from_old);

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
   * @param begin_q The begin() of the node to find.
   * @param count_q The count() of the node to find.
   * @return The found node, or NULL if not found.
   */
  const BinarySpaceTree* FindByBeginCount(size_t begin_q,
                                          size_t count_q) const;

  /**
   * Find a node in this tree by its begin and count.
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin_q The begin() of the node to find.
   * @param count_q The count() of the node to find.
   * @return The found node, or NULL if not found.
   */
  BinarySpaceTree* FindByBeginCount(size_t begin_q, size_t count_q);

  //! Return the bound object for this node.
  const Bound& bound() const;
  //! Return the bound object for this node.
  Bound& bound();

  //! Return the statistic object for this node.
  const Statistic& stat() const;
  //! Return the statistic object for this node.
  Statistic& stat();

  //! Return whether or not this node is a leaf (true if it has no children).
  bool is_leaf() const;

  /**
   * Gets the left child of this node.
   */
  BinarySpaceTree *left() const;

  /**
   * Gets the right child of this node.
   */
  BinarySpaceTree *right() const;

  /**
   * Gets the index of the beginning point of this subset.
   */
  size_t begin() const;

  /**
   * Gets the index one beyond the last index in the subset.
   */
  size_t end() const;

  /**
   * Gets the number of points in this subset.
   */
  size_t count() const;

  void Print() const;

 private:
  /**
   * Splits the current node, assigning its left and right children recursively.
   *
   * @param data Dataset which we are using.
   */
  void SplitNode(arma::mat& data);

  /**
   * Splits the current node, assigning its left and right children recursively.
   * Also returns a list of the changed indices.
   *
   * @param data Dataset which we are using.
   * @param old_from_new Vector holding permuted indices.
   */
  void SplitNode(arma::mat& data, std::vector<size_t>& old_from_new);

  /**
   * Find the index to split on for this node, given that we are splitting in
   * the given split dimension on the specified split value.
   *
   * @param data Dataset which we are using.
   * @param split_dim Dimension of dataset to split on.
   * @param split_val Value to split on, in the given split dimension.
   */
  size_t GetSplitIndex(arma::mat& data, int split_dim, double split_val);

  /**
   * Find the index to split on for this node, given that we are splitting in
   * the given split dimension on the specified split value.  Also returns a
   * list of the changed indices.
   *
   * @param data Dataset which we are using.
   * @param split_dim Dimension of dataset to split on.
   * @param split_val Value to split on, in the given split dimension.
   * @param old_from_new Vector holding permuted indices.
   */
  size_t GetSplitIndex(arma::mat& data, int split_dim, double split_val,
      std::vector<size_t>& old_from_new);

};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "binary_space_tree_impl.hpp"

#endif
