/**
 * @file kd.h
 *
 * Pointerless versions of everything needed to make different kinds of
 * trees.
 *
 * Eventually we'll have to figure out how to make these dynamic -- this
 * task ain't no trivial thing.
 */

#ifndef SUPERPAR_KD_H
#define SUPERPAR_KD_H

#include "fastlib/fastlib.h"
#include "base/otrav.h"

/**
 * A binary space partitioning tree, such as KD or ball tree, for use
 * with super-par.
 *
 * This particular tree forbids you from having more children.
 *
 * @param TBound the bounding type of each child (TODO explain interface)
 * @param TDataset the data set type
 * @param TStat extra data in the node
 *
 * @experimental
 */
template<class TBound, class TStat,
         int t_cardinality = 2>
class SpNode {
 public:
  typedef TBound Bound;
  typedef TStat Stat;
  
  enum {
    /** The root node of a tree is always at index zero. */
    ROOT_INDEX = 0
  };
  
 private:
  index_t begin_;
  index_t count_;
  
  Bound bound_;
  Stat stat_;

  index_t children_[t_cardinality];
  
  OT_DEF(SpNode) {
    OT_MY_OBJECT(begin_);
    OT_MY_OBJECT(count_);
    OT_MY_OBJECT(bound_);
    OT_MY_ARRAY(children_);
  }
  
 public:
  SpNode() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    mem::DebugPoison(children_, t_cardinality);
  }  

  ~SpNode() {
    DEBUG_ONLY(begin_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(count_ = BIG_BAD_NUMBER);
    mem::DebugPoison(children_, t_cardinality);
  }
  
  template<typename Param>
  void Init(index_t dim, const Param& param) {
    bound_.Init(dim);
    stat_.Init(param);
  }
  
  void set_range(index_t begin_in, index_t count_in) {
    DEBUG_ASSERT(begin_ == BIG_BAD_NUMBER);
    begin_ = begin_in;
    count_ = count_in;
  }

  const Bound& bound() const {
    return bound_;
  }

  Bound& bound() {
    return bound_;
  }
  
  const Stat& stat() const {
    return stat_;
  }
  
  Stat& stat() {
    return stat_;
  }

  index_t child(index_t child_number) const {
    return children_[child_number];
  }

  void set_child(index_t child_number, index_t child_index) {
    DEBUG_BOUNDS(child_number, t_cardinality);
    children_[child_number] = child_index;
  }

  void set_leaf() {
    children_[0] = -index_t(1);
  }

  bool is_leaf() const {
    return children_[0] == -index_t(1);
  }

  /**
   * Gets the index of the first point of this subset.
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
  
  /**
   * Returns the number of children of this node.
   */
  index_t cardinality() const {
    return t_cardinality;
  }
  
  void PrintSelf() const {
    printf("node: %d to %d: %d points total\n",
       begin_, begin_ + count_ - 1, count_);
  }
};

#endif
