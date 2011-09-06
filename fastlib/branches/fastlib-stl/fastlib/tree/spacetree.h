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
 * A binary space partitioning tree, such as KD or ball tree.
 *
 * This particular tree forbids you from having more children.
 *
 * @param TBound the bounding type of each child (TODO explain interface)
 * @param TDataset the data set type (must be arma::mat)
 * @param TStatistic extra data in the node
 *
 * @experimental
 */
template<typename TBound, typename TDataset,
         typename TStatistic = EmptyStatistic<TDataset> >
class BinarySpaceTree {
 public:
  typedef TBound Bound;
  typedef TDataset Dataset;
  typedef TStatistic Statistic;

 private:
  BinarySpaceTree *left_;
  BinarySpaceTree *right_;
  index_t begin_;
  index_t count_;
  Statistic stat_;
  Bound bound_;

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
  BinarySpaceTree(Dataset& data, index_t leaf_size);
  BinarySpaceTree(Dataset& data,
                  index_t leaf_size,
                  std::vector<index_t>& old_from_new);
  BinarySpaceTree(Dataset& data,
                  index_t leaf_size,
                  std::vector<index_t>& old_from_new,
                  std::vector<index_t>& new_from_old);

  BinarySpaceTree(index_t begin_in, index_t count_in);
 
  BinarySpaceTree(); 

  void Init(index_t begin_in, index_t count_in);

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
  
  /**
   * Used only when constructing the tree.
   */
  void set_children(const TDataset& data,
                    BinarySpaceTree *left_in,
                    BinarySpaceTree *right_in);

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
   */
//  void SplitNode(Dataset& data);
//  void SplitNode(Dataset& data, std::vector<index_t>& old_from_new);

//  index_t GetSplitIndex(Dataset& data, int split_dim);

};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "spacetree_impl.h"

#endif
