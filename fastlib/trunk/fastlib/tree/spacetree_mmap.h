// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file spacetree.h
 *
 * Generalized space partitioning tree.
 *
 * @experimental
 */

#ifndef TREE_SPACETREE_MMAP_H
#define TREE_SPACETREE_MMAP_H

#include "fastlib/base/base.h"
#include "fastlib/mmanager/memory_manager.h"
#include "statistic.h"

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
class BinarySpaceTreeMmap {
 public:
  typedef TBound Bound;
  typedef TDataset Dataset;
  typedef TStatistic Statistic;

 private:
  Bound bound_;
  BinarySpaceTreeMmap *left_;
  BinarySpaceTreeMmap *right_;
  index_t begin_;
  index_t count_;
  Statistic stat_;

 public:
  /*
  BinarySpaceTree() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    DEBUG_POISON_PTR(left_);
    DEBUG_POISON_PTR(right_);
  }

  ~BinarySpaceTree() {
    if (!is_leaf()) {
      delete left_;
      delete right_;
    }
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    DEBUG_POISON_PTR(left_);
    DEBUG_POISON_PTR(right_);
  }
  */
  static void* operator new(size_t size) {
    return  mmapmm::MemoryManager<false>::allocator_->Alloc<BinarySpaceTreeMmap>();
  }
  static void operator delete(void *) {
  
  }
 
  void Init(index_t begin_in, index_t count_in) {
    //DEBUG_ASSERT(begin_ == BIG_BAD_NUMBER);
    DEBUG_POISON_PTR(left_);
    DEBUG_POISON_PTR(right_);
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
  const BinarySpaceTreeMmap* FindByBeginCount(
      index_t begin_q, index_t count_q) const {
    DEBUG_ASSERT(begin_q >= begin_);
    DEBUG_ASSERT(count_q <= count_);
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
  BinarySpaceTreeMmap* FindByBeginCount(
      index_t begin_q, index_t count_q) {
    DEBUG_ASSERT(begin_q >= begin_);
    DEBUG_ASSERT(count_q <= count_);
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
  void set_children(const Dataset& data,
      BinarySpaceTreeMmap *left_in, BinarySpaceTreeMmap *right_in) {
    left_ = left_in;
    right_ = right_in;
    if (!is_leaf()) {
      stat_.Init(data, begin_, count_, left_->stat_, right_->stat_);
      DEBUG_ASSERT(count_ == left_->count_ + right_->count_);
      DEBUG_ASSERT(left_->begin_ == begin_);
      DEBUG_ASSERT(right_->begin_ == begin_ + left_->count_);
    } else {
      stat_.Init(data, begin_, count_);
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

  Statistic& stat() {
    return stat_;
  }

  bool is_leaf() const {
    return !left_;
  }

  /**
   * Gets the left branch of the tree.
   */
  BinarySpaceTreeMmap *left() const {
    // TODO: Const correctness
    return left_;
  }

  /**
   * Gets the right branch.
   */
  BinarySpaceTreeMmap *right() const {
    // TODO: Const correctness
    return right_;
  }

  /**
   * Gets the index of the begin point of this subset.
   */
  index_t begin() const {
    return begin_;
  }

  /**
   * Gets the index one beyond the last index in the series.
   */
  index_t end() const {
    return begin_ + count_;
  }
  
  /**
   * Gets the number of points in this subset.
   */
  index_t count() const {
    return count_;
  }
  
  void Print() const {
    printf("node: %d to %d: %d points total\n",
       begin_, begin_ + count_ - 1, count_);
    if (!is_leaf()) {
      left_->Print();
      right_->Print();
    }
  }

};

#endif
