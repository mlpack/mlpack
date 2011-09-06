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
#include "kdtree_impl.h"
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
    bound_(data.n_rows) {
  // Make the vector of dimensions to split on.  Don't use linspace because it
  // fails with just one dimension...
  arma::uvec dims(data.n_rows);
  for (index_t i = 0; i < data.n_rows; i++)
    dims[i] = i;

  // Now, do the actual splitting of this node.
  tree_kdtree_private::SelectSplitKdTreeMidpoint(data, dims, this,
      leaf_size /* leaf_size needs to be parameterizable */);
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
    bound_(data.n_rows) {
  // Initialize old_from_new correctly.
  old_from_new.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Make the vector of dimensions to split on.  Don't use linspace because it
  // fails with just one dimension...
  arma::uvec dims(data.n_rows);
  for (index_t i = 0; i < data.n_rows; i++)
    dims[i] = i;

  // Now do the actual splitting.
  tree_kdtree_private::SelectSplitKdTreeMidpoint(data, dims, this,
      leaf_size /* leaf_size needs to be parameterizable */, old_from_new);
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
    bound_(data.n_rows) {
  // Initialize the old_from_new vector correctly.
  old_from_new.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Make the vector of dimensions to split on.  Don't use linspace because it
  // fails with just one dimension...
  arma::uvec dims(data.n_rows);
  for (index_t i = 0; i < data.n_rows; i++)
    dims[i] = i;

  // Now do the actual splitting.
  tree_kdtree_private::SelectSplitKdTreeMidpoint(data, dims, this,
      leaf_size /* leaf_size needs to be parameterizable */, old_from_new);

  // Map the new_from_old indices correctly.
  new_from_old.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    new_from_old[old_from_new[i]] = i;
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    index_t begin_in,
    index_t count_in) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in) {
  // Nothing else to do.
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree() :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(0) {
  // Nothing to do.
}

template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::Init(index_t begin_in,
                                                         index_t count_in) {
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
  
/**
 * Used only when constructing the tree.
 */
template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::set_children(
    const TDataset& data,
    BinarySpaceTree *left_in,
    BinarySpaceTree *right_in) {
  left_ = left_in;
  right_ = right_in;
  if (!is_leaf()) {
    stat_.Init(data, begin_, count_, left_->stat_, right_->stat_);
    mlpack::IO::Assert(count_ == left_->count_ + right_->count_);
    mlpack::IO::Assert(left_->begin_ == begin_);
    mlpack::IO::Assert(right_->begin_ == begin_ + left_->count_);
  } else {
    stat_.Init(data, begin_, count_);
  }
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
/*
template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::SplitNode(Dataset& data) {
  // First, check if we need to split at all.
  if (count_ < leaf_size)
    return; // We can't split this.

  // Figure out which dimension to split on.
  index_t split_dim = dataset.n_rows; // Indicate invalid by max_dim + 1.
  double max_width = -1;

  // This should be a single function for TBound.
  // We need to expand the bounds of this node properly.
  for (index_t i = begin_; i < (begin_ + count_); i++)
    bounds |= data.unsafe_col(i);

  // 
}

template<typename TBound, typename TDataset, typename TStatistic>
void BinarySpaceTree<TBound, TDataset, TStatistic>::SplitNode(
    Dataset& data,
    std::vector<index_t>& old_from_new) {

}
*/
}; // namespace tree
}; // namespace mlpack

#endif
