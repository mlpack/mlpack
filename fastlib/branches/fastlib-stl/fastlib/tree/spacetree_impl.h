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

namespace mlpack {
namespace tree {

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(Dataset& data,
    index_t leaf_size) :
    begin_(0), /* This root node starts at index 0, */
    count_(data.n_cols), /* and spans all of the dataset. */
    bound_(data.n_rows) {
  // Make the vector of dimensions to split on.
  arma::uvec dims = arma::linspace<arma::uvec>(0, data.n_rows - 1, data.n_rows);

  // First, we have to set the bound of this node correctly.
  tree_kdtree_private::SelectFindBoundFromMatrix(data, dims, begin_, count_,
      &bound_);

  // Now, do the actual splitting of this node.
  tree_kdtree_private::SelectSplitKdTreeMidpoint(data, dims, this,
      leaf_size /* leaf_size needs to be parameterizable */);
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree(
    Dataset& data,
    index_t leaf_size,
    std::vector<index_t>& old_from_new) :
    begin_(0),
    count_(data.n_cols),
    bound_(data.n_rows) {
  // Initialize old_from_new correctly.
  old_from_new.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Make the vector of dimensions to split on.
  arma::uvec dims = arma::linspace<arma::uvec>(0, data.n_rows - 1, data.n_rows);

  // Set the bound of this node correctly.
  tree_kdtree_private::SelectFindBoundFromMatrix(data, dims, begin_, count_,
      &bound_);

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
    begin_(0),
    count_(data.n_cols),
    bound_(data.n_rows) {
  // Initialize the old_from_new vector correctly.
  old_from_new.resize(data.n_cols);
  for (index_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Make the vector of dimensions to split on.
  arma::uvec dims = arma::linspace<arma::uvec>(0, data.n_rows - 1, data.n_rows);

  // Set the bound of this node correctly.
  tree_kdtree_private::SelectFindBoundFromMatrix(data, dims, begin_, count_,
      &bound_);

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
    begin_(begin_in),
    count_(count_in) {
  // Nothing else to do.
}

template<typename TBound, typename TDataset, typename TStatistic>
BinarySpaceTree<TBound, TDataset, TStatistic>::BinarySpaceTree() :
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

  DEBUG_ASSERT(begin_q >= begin_);
  DEBUG_ASSERT(count_q <= count_);
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

  DEBUG_ASSERT(begin_q >= begin_);
  DEBUG_ASSERT(count_q <= count_);
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
    DEBUG_ASSERT(count_ == left_->count_ + right_->count_);
    DEBUG_ASSERT(left_->begin_ == begin_);
    DEBUG_ASSERT(right_->begin_ == begin_ + left_->count_);
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

}; // namespace tree
}; // namespace mlpack

#endif
