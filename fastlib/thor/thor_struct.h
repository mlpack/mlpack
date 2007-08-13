/**
 * @file thorstruct.h
 *
 * The most basic THOR tree structures.
 */

#ifndef THOR_THORSTRUCT_H
#define THOR_THORSTRUCT_H

#include "la/matrix.h"
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
class ThorNode {
 public:
  typedef TBound Bound;
  typedef TStat Stat;
  enum { CARDINALITY = t_cardinality };

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

  OT_DEF_BASIC(ThorNode) {
    OT_MY_OBJECT(begin_);
    OT_MY_OBJECT(count_);
    OT_MY_OBJECT(bound_);
    OT_MY_OBJECT(stat_);
    OT_MY_ARRAY(children_);
  }

 public:
  template<typename Param>
  void Init(index_t dim, const Param& param) {
    bound_.Init(dim);
    stat_.Init(param);
  }

  void set_range(index_t begin_in, index_t count_in) {
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

  index_t child(int child_number) const {
    return children_[child_number];
  }

  void set_child(int child_number, index_t child_index) {
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

/**
 * A skeleton of a huge tree.
 *
 * The skeleton is just a pointer-type version of ThorNode, though it's too
 * space-inefficient to use as a tree itself since it wastes space on
 * indices.  Instead, this begins as the root of a cached tree and can be
 * expanded on demand.
 *
 * This can be associated with extra bookkeeping information, useful for
 * instance for .
 */
template<typename TNode, typename TInfo>
class ThorSkeletonNode {
 public:
  typedef TNode Node;
  typedef TInfo Info;

 private:
  index_t index_;
  index_t end_index_;
  Info info_;
  Node node_;
  ThorSkeletonNode *parent_;
  ThorSkeletonNode *children_[Node::CARDINALITY];

  OT_DEF(ThorSkeletonNode) {
    OT_MY_OBJECT(index_);
    OT_MY_OBJECT(info_);
    OT_MY_OBJECT(node_);
    for (int k = 0; k < Node::CARDINALITY; k++) {
      OT_PTR_NULLABLE(children_[k]);
    }
  }

  OT_FIX(ThorSkeletonNode) {
    parent_ = NULL;

    for (int k = 0; k < Node::CARDINALITY; k++) {
      ThorSkeletonNode *c = children_[k];
      if (c) {
        c->parent_ = this;
      }
    }
  }

 public:
  /**
   * Constructs this.
   *
   * We're using constructors because it lets us use primitives.
   *
   * @param info_in the info object to use
   * @param array where to get tree information from
   * @param index_in the index of the node in the tree
   * @param parent_in the parent node, or NULL if this is the root
   */
  ThorSkeletonNode(const Info& info_in, CacheArray<Node> *array,
      index_t node_index_in, index_t end_index_in,
      ThorSkeletonNode *parent_in = NULL)
        : index_(node_index_in)
        , end_index_(end_index_in)
        , info_(info_in)
        , node_(*array->StartRead(node_index_in))
        , parent_(parent_in) {
    array->StopRead(node_index_in);
    for (int k = 0; k < Node::CARDINALITY; k++) {
      children_[k] = NULL;
    }
  }

  Info& info() {
    return info_;
  }
  const Info& info() const {
    return info_;
  }
  Node& node() {
    return node_;
  }
  const Node& node() const {
    return node_;
  }
  index_t index() const {
    return index_;
  }
  bool is_leaf() const {
    return node_.is_leaf();;
  }
  ThorSkeletonNode *parent() const {
    return parent_;
  }
  bool is_root() const {
    return parent_ == NULL;
  }
  index_t end_index() const {
    return end_index_;
  }
  index_t count() const {
    return node_.count();
  }

  void set_child(int k, ThorSkeletonNode *child) {	
    DEBUG_ASSERT(child->parent_ == NULL);
    DEBUG_ASSERT(children_[k] == NULL);
    DEBUG_ASSERT(node_.child(k) == child->index());
    child->parent_ = this;
    children_[k] = child;
  }

  /**
   * Gets a child node, copying over info from parent.
   *
   * @param array the array to read information from if the node is not
   *        already in the skeleton tree
   * @param k child number (k.e. 0 for left and 1 for right)
   */
  ThorSkeletonNode *GetChild(CacheArray<Node> *array, int k) {
    if (children_[k] == NULL && !node_.is_leaf()) {
      index_t child_end_index; // compute end index based on pre-order
      if (k + 1 == Node::CARDINALITY) {
        child_end_index = end_index_;
      } else {
        child_end_index = node_.child(k+1);
      }
      children_[k] = new ThorSkeletonNode(info_, array, node_.child(k),
          child_end_index, this);
    }
    return children_[k];
  }

  /**
   * Gets a child of the node.
   *
   * This may return NULL even if the node isn't a leaf!  This is because
   * this tree is only a skeleton, and unexplored parts of the tree are
   * NULL.  Use is_leaf(), which in turn calls node().is_leaf(), to check
   * for leafness.
   */
  ThorSkeletonNode *child(int k) const {
    return children_[k];
  }
  /**
   * Returns whether all children exist statically.
   */
  bool is_complete() const {
    for (int k = 0; k < Node::CARDINALITY; k++) {
      if (!children_[k]) {
        return false;
      }
    }
    return true;
  }
};

#endif
