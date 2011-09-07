/**
 * @file spacetree.h
 *
 * Generalized space partitioning tree.
 *
 * @experimental
 */

#ifndef TREE_SPACETREE_H
#define TREE_SPACETREE_H

#include "../base/common.h"
#include "statistic.h"

#include <armadillo>

namespace mlpack {
namespace tree {

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
  BinarySpaceTree *left_; //< The left child node.
  BinarySpaceTree *right_; //< The right child node.
  index_t begin_; //< The first point in the dataset contained in this node.
  index_t count_; //< The count of points in the dataset contained in this node.
  Bound bound_; //< The bound object for this node.
  Statistic stat_; //< The extra data contained in the node.

 public:
  /***
   * Construct this as the head node of a binary space tree using the given
   * dataset.  This will modify the ordering of the points in the dataset!
   *
   * Optionally, pass in vectors which represent a mapping from the old
   * dataset's point ordering to the new ordering, and vice versa.
   *
   * @param data Dataset to create tree from.
   * @param leaf_size Leaf size of the tree.
   * @param old_from_new Vector which will be filled with the old positions for
   *     each new point.
   * @param new_from_old Vector which will be filled with the new positions for
   *     each old point.
   */
  BinarySpaceTree(arma::mat& data, index_t leaf_size);
  BinarySpaceTree(arma::mat& data,
                  index_t leaf_size,
                  std::vector<index_t>& old_from_new);
  BinarySpaceTree(arma::mat& data,
                  index_t leaf_size,
                  std::vector<index_t>& old_from_new,
                  std::vector<index_t>& new_from_old);

  BinarySpaceTree(arma::mat& data,
                  index_t leaf_size,
                  index_t begin_in,
                  index_t count_in);
  BinarySpaceTree(arma::mat& data,
                  index_t leaf_size,
                  index_t begin_in,
                  index_t count_in,
                  std::vector<index_t>& old_from_new);
  BinarySpaceTree(arma::mat& data,
                  index_t leaf_size,
                  index_t begin_in,
                  index_t count_in,
                  std::vector<index_t>& old_from_new,
                  std::vector<index_t>& new_from_old);
 
  BinarySpaceTree();

  /***
   * Deletes this node, deallocating the memory for the children and calling
   * their destructors in turn.  This will invalidate any pointers or references
   * to any nodes which are children of this one.
   */
  ~BinarySpaceTree();

  /**
   * Find a node in this tree by its begin and count.
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin_q the begin() of the node to find
   * @param count_q the count() of the node to find
   * @return the found node, or NULL
   */
  const BinarySpaceTree* FindByBeginCount(index_t begin_q,
                                          index_t count_q) const;
  
  /**
   * Find a node in this tree by its begin and count (const).
   *
   * Every node is uniquely identified by these two numbers.
   * This is useful for communicating position over the network,
   * when pointers would be invalid.
   *
   * @param begin_q the begin() of the node to find
   * @param count_q the count() of the node to find
   * @return the found node, or NULL
   */
  BinarySpaceTree* FindByBeginCount(index_t begin_q, index_t count_q);
  
  // TODO: Not const correct
  
  const Bound& bound() const;
  Bound& bound();

  const Statistic& stat() const;
  Statistic& stat();

  bool is_leaf() const;

  /**
   * Gets the left branch of the tree.
   */
  BinarySpaceTree *left() const;

  /**
   * Gets the right branch.
   */
  BinarySpaceTree *right() const;

  /**
   * Gets the index of the begin point of this subset.
   */
  index_t begin() const;

  /**
   * Gets the index one beyond the last index in the series.
   */
  index_t end() const;
  
  /**
   * Gets the number of points in this subset.
   */
  index_t count() const;
  
  void Print() const;

 private:

  /***
   * Splits the current node, assigning its left and right children recursively.
   *
   * Optionally, return a list of the changed indices.
   *
   * @param data Dataset which we are using.
   * @param leaf_size Leaf size to split with.
   * @param old_from_new Vector holding permuted indices.
   */
  void SplitNode(arma::mat& data, index_t leaf_size);
  void SplitNode(arma::mat& data, index_t leaf_size,
      std::vector<index_t>& old_from_new);

  /***
   * Find the index to split on for this node, given that we are splitting in
   * the given split dimension on the specified split value.
   *
   * Optionally, return a list of the changed indices.
   *
   * @param data Dataset which we are using.
   * @param split_dim Dimension of dataset to split on.
   * @param split_val Value to split on, in the given split dimension.
   * @param old_from_new Vector holding permuted indices.
   */
  index_t GetSplitIndex(arma::mat& data, int split_dim, double split_val);
  index_t GetSplitIndex(arma::mat& data, int split_dim, double split_val,
      std::vector<index_t>& old_from_new);

};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "spacetree_impl.h"

#endif
