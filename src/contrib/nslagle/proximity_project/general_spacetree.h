// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTCLIN
/**
 * @file general_spacetree.h
 *
 * Generalized space partitioning tree.
 *
 * @experimental
 */

#ifndef TREE_GENERAL_SPACETREE_H
#define TREE_GENERAL_SPACETREE_H

#include <mlpack/core.h>
#include "mlpack/core/tree/statistic.hpp"

#define _lineStr(x) # x
#define lineStr(x) _lineStr(x)
#define assertMsg __FILE__":" lineStr(__LINE__)

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
         class TStatistic = EmptyStatistic >
class GeneralBinarySpaceTree {
  public:
  typedef TBound Bound;
  typedef TDataset Dataset;
  typedef TStatistic Statistic;

  Bound bound_;
  GeneralBinarySpaceTree *left_;
  GeneralBinarySpaceTree *right_;
  size_t begin_;
  size_t count_;
  Statistic stat_;

  public:
  GeneralBinarySpaceTree(size_t dimension) {
    bound_ = Bound (dimension);
    //DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    //DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    //DEBUG_POISON_PTR(left_);
    //DEBUG_POISON_PTR(right_);
  }

  ~GeneralBinarySpaceTree() {
    if (!is_leaf()) {
      delete left_;
      delete right_;
    }
    //DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    //DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    //DEBUG_POISON_PTR(left_);
    //DEBUG_POISON_PTR(right_);
  }

  void Init(size_t begin_in, size_t count_in) {
    //mlpack::Log::Info << "the beginning " << begin_ << std::endl;
    //mlpack::Log::Assert(begin_ == ~((size_t)0), assertMsg);
    left_ = NULL;
    right_ = NULL;
    begin_ = begin_in;
    count_ = count_in;
  }

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
  const GeneralBinarySpaceTree* FindByBeginCount
  (size_t begin_q, size_t count_q) const {
    mlpack::Log::Assert(begin_q >= begin_, assertMsg);
    mlpack::Log::Assert(count_q <= count_, assertMsg);
    if (begin_ == begin_q && count_ == count_q) {
      return this;
    } else if (unlikely(is_leaf())) {
      return NULL;
    } else if (begin_q < right_->begin_) {
      return left_->FindByBeginCount(begin_q, count_q);
    } else {
      return right_->FindByBeginCount(begin_q, count_q);
    }
  }

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
  GeneralBinarySpaceTree* FindByBeginCount
  (size_t begin_q, size_t count_q) {
    mlpack::Log::Assert(begin_q >= begin_, assertMsg);
    mlpack::Log::Assert(count_q <= count_, assertMsg);
    if (begin_ == begin_q && count_ == count_q) {
      return this;
    } else if (unlikely(is_leaf())) {
      return NULL;
    } else if (begin_q < right_->begin_) {
      return left_->FindByBeginCount(begin_q, count_q);
    } else {
      return right_->FindByBeginCount(begin_q, count_q);
    }
  }

  // TODO: Not const correct

  /**
   * Used only when constructing the tree.
   */
  void set_children
  (const Dataset& data, GeneralBinarySpaceTree *left_in,
   GeneralBinarySpaceTree *right_in) {
     left_ = left_in;
     right_ = right_in;
     if (is_leaf()) {
       stat_.Init(data, begin_, count_);
     }
     else {
       stat_.Init(data, begin_, count_, left_->stat_, right_->stat_);
     }
   }

  const Bound& bound() const {
    return bound_;
  }

  Bound& bound() {
    return bound_;
  }

  const Statistic& stat() const {
    return stat_;
  }

  void setBound (Bound b)
  {
    bound_ = b;
  }


  Statistic& stat() {
    return stat_;
  }

  bool is_leaf() const {
    return !left_;
  }

  /**
   * Gets the left branch of the tree.
   */
  GeneralBinarySpaceTree *left() const {
    // TODO: Const correctness
    return left_;
  }

  /**
   * Gets the right branch.
   */
  GeneralBinarySpaceTree *right() const {
    // TODO: Const correctness
    return right_;
  }

  /**
   * Gets the index of the begin point of this subset.
   */
  size_t begin() const {
    return begin_;
  }

  /**
   * Gets the index one beyond the last index in the series.
   */
  size_t end() const {
    return begin_ + count_;
  }

  /**
   * Gets the number of points in this subset.
   */
  size_t count() const {
    return count_;
  }

  void Print() const {
    if (!is_leaf()) {
      printf("internal node: %d to %d: %d points total\n",
       begin_, begin_ + count_ - 1, count_);
    }
    else {
      printf("leaf node: %d to %d: %d points total\n",
       begin_, begin_ + count_ - 1, count_);
    }

    if (!is_leaf()) {
      left_->Print();
      right_->Print();
    }
  }

};

#endif
