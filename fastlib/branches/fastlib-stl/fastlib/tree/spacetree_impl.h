/**
 * @file spacetree_impl.h
 *
 * Implementation of generalized space partitioning tree.
 *
 * @experimental
 */

#ifndef TREE_SPACETREE_IMPL_H
#define TREE_SPACETREE_IMPL_H

// In case it wasn't included already for some reason.
#include "spacetree.h"
#include "../fx/io.h"

namespace mlpack {
namespace tree {

// Each of these overloads is kept as a separate function to keep the overhead
// from the two std::vectors out, if possible.
template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(Dataset& data,
    index_t leaf_size) :
    left_(NULL),
    right_(NULL),
    begin_(0), /* This root node starts at index 0, */
    count_(data.n_cols), /* and spans all of the dataset. */
    bound_(data.n_rows),
    stat_() {
  // Do the actual splitting of this node.
  SplitNode(data, leaf_size);
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    Dataset& data,
    index_t leaf_size,
    std::vector<index_t>& old_from_new) :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(data.n_cols),
    bound_(data.n_rows),
    stat_() {
  // Initialize old_from_new correctly.
  old_from_new.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitNode(data, leaf_size, old_from_new);
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    Dataset& data,
    index_t leaf_size,
    std::vector<index_t>& old_from_new,
    std::vector<index_t>& new_from_old) :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(data.n_cols),
    bound_(data.n_rows),
    stat_() {
  // Initialize the old_from_new vector correctly.
  old_from_new.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitNode(data, leaf_size, old_from_new);

  // Map the new_from_old indices correctly.
  new_from_old.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    new_from_old[old_from_new[i]] = i;

  IO::Debug << "old from new is " << std::endl;
  for (index_t i = 0; i < data.n_cols; i++) {
    IO::Debug << old_from_new[i] << " ";
  }
  IO::Debug << std::endl;
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    Dataset& data,
    index_t leaf_size,
    index_t begin_in,
    index_t count_in) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in),
    bound_(data.n_rows),
    stat_() {
  // Perform the actual splitting.
  SplitNode(data, leaf_size);
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    Dataset& data,
    index_t leaf_size,
    index_t begin_in,
    index_t count_in,
    std::vector<index_t>& old_from_new) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in),
    bound_(data.n_rows),
    stat_() {
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  assert(old_from_new.size() == data.n_cols);

  // Perform the actual splitting.
  SplitNode(data, leaf_size, old_from_new);
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    Dataset& data,
    index_t leaf_size,
    index_t begin_in,
    index_t count_in,
    std::vector<index_t>& old_from_new,
    std::vector<index_t>& new_from_old) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in),
    bound_(data.n_rows),
    stat_() {
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  assert(old_from_new.size() == data.n_cols);

  // Perform the actual splitting.
  SplitNode(data, leaf_size, old_from_new);

  // Map the new_from_old indices correctly.
  new_from_old.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    new_from_old[old_from_new[i]] = i;
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree() :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(0),
    bound_(),
    stat_() {
  // Nothing to do.
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
template<typename TBound, typename TDataset, typename TStatistic>
const BinarySpaceTree<TBound, TDataset, TStatistic>*
BinarySpaceTree<TBound, TDataset, TStatistic>::FindByBeginCount(
    index_t begin_q,
    index_t count_q) const {

  mlpack::IO::Assert(begin_q >= begin_);
  mlpack::IO::Assert(count_q <= count_);
  if (begin_ == begin_q && count_ == count_q)
    return this;
  else if (is_leaf())
    return NULL;
  else if (begin_q < right_->begin_)
    return left_->FindByBeginCount(begin_q, count_q);
  else
    return right_->FindByBeginCount(begin_q, count_q);
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
template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>*
BinarySpaceTree<TBound, TDataset, TStatistic>::FindByBeginCount(
    index_t begin_q,
    index_t count_q) {

  mlpack::IO::Assert(begin_q >= begin_);
  mlpack::IO::Assert(count_q <= count_);
  if (begin_ == begin_q && count_ == count_q)
    return this;
  else if (unlikely(is_leaf()))
    return NULL;
  else if (begin_q < right_->begin_)
    return left_->FindByBeginCount(begin_q, count_q);
  else
    return right_->FindByBeginCount(begin_q, count_q);
}
  
template<typename TBound, typename TDataset, typename TStatistic>
const TBound& BinarySpaceTree<TBound, TDataset, TStatistic>::bound() const {
  return bound_;
}

template<typename TBound, typename TDataset, typename TStatistic>
TBound& BinarySpaceTree<TBound, TDataset, TStatistic>::bound() {
  return bound_;
}

template<typename TBound, typename TDataset, typename TStatistic>
const TStatistic& BinarySpaceTree<TBound, TDataset, TStatistic>::stat() const {
  return stat_;
}

template<typename TBound, typename TDataset, typename TStatistic>
TStatistic& BinarySpaceTree<TBound, TDataset, TStatistic>::stat() {
  return stat_;
}

template<typename TBound, typename TDataset, typename TStatistic>
bool BinarySpaceTree<TBound, TDataset, TStatistic>::is_leaf() const {
  return !left_;
}

/**
 * Gets the left branch of the tree.
 */
template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>* 
BinarySpaceTree<TBound, TDataset, TStatistic>::left() const {
  // TODO: Const correctness
  return left_;
}

/**
 * Gets the right branch.
 */
template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>*
BinarySpaceTree<TBound, TDataset, TStatistic>::right() const {
  // TODO: Const correctness
  return right_;
}

/**
 * Gets the index of the begin point of this subset.
 */
template<typename TBound, typename TDataset, typename TStatistic>
index_t BinarySpaceTree<TBound, TDataset, TStatistic>::begin() const {
  return begin_;
}

/**
 * Gets the index one beyond the last index in the series.
 */
template<typename TBound, typename TDataset, typename TStatistic>
index_t BinarySpaceTree<TBound, TDataset, TStatistic>::end() const {
  return begin_ + count_;
}
  
/**
 * Gets the number of points in this subset.
 */
template<typename TBound, typename TDataset, typename TStatistic>
index_t BinarySpaceTree<TBound, TDataset, TStatistic>::count() const {
  return count_;
}
  
template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::Print() const {
  printf("node: %d to %d: %d points total\n",
     begin_, begin_ + count_ - 1, count_);
  if (!is_leaf()) {
    left_->Print();
    right_->Print();
  }
}

template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::SplitNode(Dataset& data,
    index_t leaf_size) {
  // This should be a single function for TBound.
  // We need to expand the bounds of this node properly.
  for (index_t i = begin_; i < (begin_ + count_); i++)
    bound_ |= data.unsafe_col(i);

  // Now, check if we need to split at all.
  if (count_ <= leaf_size)
    return; // We can't split this.

  // Figure out which dimension to split on.
  index_t split_dim = data.n_rows; // Indicate invalid by max_dim + 1.
  double max_width = -1;

  // Find the split dimension.
  for (index_t d = 0; d < data.n_rows; d++) {
    double width = bound_[d].width();

    if (width > max_width) {
      max_width = width;
      split_dim = d;
    }
  }

  // Split in the middle of that dimension.
  double split_val = bound_[split_dim].mid();

  if (max_width == 0) // All these points are the same.  We can't split.
    return;

  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension split_dim less than or equal to split_val are on
  // the left of split_col, and points with value in dimension split_dim greater
  // than split_val are on the right side of split_col.
  index_t split_col = GetSplitIndex(data, split_dim, split_val);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left_ = new BinarySpaceTree<TBound, TDataset, TStatistic>(data, leaf_size,
      begin_, split_col - begin_);
  right_ = new BinarySpaceTree<TBound, TDataset, TStatistic>(data, leaf_size,
      split_col, begin_ + count_ - split_col);
}

template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::SplitNode(
    Dataset& data,
    index_t leaf_size,
    std::vector<index_t>& old_from_new) {
  // This should be a single function for TBound.
  // We need to expand the bounds of this node properly.
  for (index_t i = begin_; i < (begin_ + count_); i++)
    bound_ |= data.unsafe_col(i);

  // First, check if we need to split at all.
  if (count_ <= leaf_size)
    return; // We can't split this.

  // Figure out which dimension to split on.
  index_t split_dim = data.n_rows; // Indicate invalid by max_dim + 1.
  double max_width = -1;

  // Find the split dimension.
  for (index_t d = 0; d < data.n_rows; d++) {
    double width = bound_[d].width();

    if (width > max_width) {
      max_width = width;
      split_dim = d;
    }
  }

  // Split in the middle of that dimension.
  double split_val = bound_[split_dim].mid();

  if (max_width == 0) // All these points are the same.  We can't split.
    return;

  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension split_dim less than or equal to split_val are on
  // the left of split_col, and points with value in dimension split_dim greater
  // than split_val are on the right side of split_col.
  index_t split_col = GetSplitIndex(data, split_dim, split_val, old_from_new);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left_ = new BinarySpaceTree<TBound, TDataset, TStatistic>(data, leaf_size,
      begin_, split_col - begin_, old_from_new);
  right_ = new BinarySpaceTree<TBound, TDataset, TStatistic>(data, leaf_size,
      split_col, begin_ + count_ - split_col, old_from_new);
}

template<typename TBound, typename TDataset, typename TStatistic>
index_t BinarySpaceTree<TBound, TDataset, TStatistic>::GetSplitIndex(
    Dataset& data,
    int split_dim,
    double split_val) {
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points less than
  // split_val should be on the left side of the matrix, and the points greater
  // than split_val should be on the right side of the matrix.
  index_t left = begin_;
  index_t right = begin_ + count_ - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((data(split_dim, left) < split_val) && (left <= right))
    left++;
  while ((data(split_dim, right) >= split_val) && (left <= right))
    right--;

  while(left <= right) {
    // Swap columns.
    data.swap_cols(left, right);

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while ((data(split_dim, left) < split_val) && (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((data(split_dim, right) >= split_val) && (left <= right))
      right--;
  }

  assert(left == right + 1);

  return left;
}

template<typename TBound, typename TDataset, typename TStatistic>
index_t BinarySpaceTree<TBound, TDataset, TStatistic>::GetSplitIndex(
    Dataset& data,
    int split_dim,
    double split_val,
    std::vector<index_t>& old_from_new) {
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points less than
  // split_val should be on the left side of the matrix, and the points greater
  // than split_val should be on the right side of the matrix.
  index_t left = begin_;
  index_t right = begin_ + count_ -1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle. 
  while ((data(split_dim, left) < split_val) && (left <= right))
    left++;
  while ((data(split_dim, right) >= split_val) && (left <= right))
    right--;

  while(left <= right) {
    // Swap columns.
    data.swap_cols(left, right);

    // Update the indices for what we changed.
    index_t t = old_from_new[left];
    old_from_new[left] = old_from_new[right];
    old_from_new[right] = t;

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while ((data(split_dim, left) < split_val) && (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((data(split_dim, right) >= split_val) && (left <= right))
      right--;
  }

  assert(left == right + 1);

  return left;
}

}; // namespace tree
}; // namespace mlpack

#endif
