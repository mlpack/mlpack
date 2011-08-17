/**
 * @file spacetree.h
 *
 * Generalized space partitioning tree.
 *
 * @experimental
 */

#ifndef TREE_SPACETREE_H
#define TREE_SPACETREE_H

#include "../base/base.h"
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
 * @param TDataset the data set type
 * @param TStatistic extra data in the node
 *
 * @experimental
 */
template<class TBound,
         class TDataset,
         class TStatistic = EmptyStatistic<TDataset> >
class BinarySpaceTree {
 public:
  typedef TBound Bound;
  typedef TDataset Dataset;
  typedef TStatistic Statistic;

 private:
  Bound bound_;
  BinarySpaceTree *left_;
  BinarySpaceTree *right_;
  index_t begin_;
  index_t count_;
  Statistic stat_;

 public:
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

};

}; // namespace tree
}; // namespace mlpack

// Include implementation.
#include "spacetree_impl.h"

#endif
