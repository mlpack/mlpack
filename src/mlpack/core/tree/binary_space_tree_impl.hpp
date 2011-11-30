/**
 * @file binary_space_tree_impl.hpp
 *
 * Implementation of generalized space partitioning tree.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "binary_space_tree.hpp"

#include <mlpack/core/io/cli.hpp>
#include <mlpack/core/io/log.hpp>

namespace mlpack {
namespace tree {

// Each of these overloads is kept as a separate function to keep the overhead
// from the two std::vectors out, if possible.
template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree(arma::mat& data) :
    left_(NULL),
    right_(NULL),
    begin_(0), /* This root node starts at index 0, */
    count_(data.n_cols), /* and spans all of the dataset. */
    bound_(data.n_rows),
    stat_()
{
  // Do the actual splitting of this node.
  SplitNode(data);
}

template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree(
    arma::mat& data,
    std::vector<size_t>& old_from_new) :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(data.n_cols),
    bound_(data.n_rows),
    stat_()
{
  // Initialize old_from_new correctly.
  old_from_new.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitNode(data, old_from_new);
}

template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree(
    arma::mat& data,
    std::vector<size_t>& old_from_new,
    std::vector<size_t>& new_from_old) :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(data.n_cols),
    bound_(data.n_rows),
    stat_()
{
  // Initialize the old_from_new vector correctly.
  old_from_new.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    old_from_new[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitNode(data, old_from_new);

  // Map the new_from_old indices correctly.
  new_from_old.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    new_from_old[old_from_new[i]] = i;
}

template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree(
    arma::mat& data,
    size_t begin_in,
    size_t count_in) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in),
    bound_(data.n_rows),
    stat_()
{
  // Perform the actual splitting.
  SplitNode(data);
}

template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree(
    arma::mat& data,
    size_t begin_in,
    size_t count_in,
    std::vector<size_t>& old_from_new) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in),
    bound_(data.n_rows),
    stat_()
{
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  Log::Assert(old_from_new.size() == data.n_cols);

  // Perform the actual splitting.
  SplitNode(data, old_from_new);
}

template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree(
    arma::mat& data,
    size_t begin_in,
    size_t count_in,
    std::vector<size_t>& old_from_new,
    std::vector<size_t>& new_from_old) :
    left_(NULL),
    right_(NULL),
    begin_(begin_in),
    count_(count_in),
    bound_(data.n_rows),
    stat_()
{
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  Log::Assert(old_from_new.size() == data.n_cols);

  // Perform the actual splitting.
  SplitNode(data, old_from_new);

  // Map the new_from_old indices correctly.
  new_from_old.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    new_from_old[old_from_new[i]] = i;
}

template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::BinarySpaceTree() :
    left_(NULL),
    right_(NULL),
    begin_(0),
    count_(0),
    bound_(),
    stat_()
{
  // Nothing to do.
}

/**
 * Deletes this node, deallocating the memory for the children and calling their
 * destructors in turn.  This will invalidate any pointers or references to any
 * nodes which are children of this one.
 */
template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>::~BinarySpaceTree()
{
  if (left_)
    delete left_;
  if (right_)
    delete right_;
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
template<typename Bound, typename Statistic>
const BinarySpaceTree<Bound, Statistic>*
BinarySpaceTree<Bound, Statistic>::FindByBeginCount(size_t begin_q,
                                                    size_t count_q) const
{
  mlpack::Log::Assert(begin_q >= begin_);
  mlpack::Log::Assert(count_q <= count_);
  if (begin_ == begin_q && count_ == count_q)
    return this;
  else if (is_leaf())
    return NULL;
  else if (begin_q < left_->end_)
    return left_->FindByBeginCount(begin_q, count_q);
  else if (right_)
    return right_->FindByBeginCount(begin_q, count_q);
  else
    return NULL;
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
template<typename Bound, typename Statistic>
BinarySpaceTree<Bound, Statistic>*
BinarySpaceTree<Bound, Statistic>::FindByBeginCount(size_t begin_q,
                                                    size_t count_q)
{
  mlpack::Log::Assert(begin_q >= begin_);
  mlpack::Log::Assert(count_q <= count_);
  if (begin_ == begin_q && count_ == count_q)
    return this;
  else if (is_leaf())
    return NULL;
  else if (begin_q < left_->end_)
    return left_->FindByBeginCount(begin_q, count_q);
  else if (right_)
    return right_->FindByBeginCount(begin_q, count_q);
  else
    return NULL;
}

template<typename Bound, typename Statistic>
size_t BinarySpaceTree<Bound, Statistic>::ExtendTree(size_t level)
{
  --level;
  /* return the number of nodes duplicated */
  size_t nodesDuplicated = 0;
  if (level > 0)
  {
    if (!left_)
    {
      left_ = CopyMe();
      ++nodesDuplicated;
    }
    nodesDuplicated += left_->ExtendTree(level);
    if (right_)
    {
      nodesDuplicated += right_->ExtendTree(level);
    }
  }
  return nodesDuplicated;
}

/* TODO: we can likely calculate this earlier, then store the
 *   result in a private member variable; for now, we can
 *   just calculate as needed...
 *
 *   Also, perhaps we should rewrite these recursive functions
 *     to avoid exceeding the stack limit
 */

template<typename Bound, typename Statistic>
size_t BinarySpaceTree<Bound, Statistic>::TreeSize() const
{
  size_t size = 1;
  if (is_leaf())
  {
    return size;
  }
  else
  {
    size += this->left()->TreeSize();
    if (right_)
    {
      size += this->right()->TreeSize();
    }
  }
  return size;
}

template<typename Bound, typename Statistic>
size_t BinarySpaceTree<Bound, Statistic>::TreeDepth() const
{
  size_t levels = 1;
  if (is_leaf())
  {
    return levels;
  }
  else
  {
    size_t rightLevels = 0;
    if (right_)
    {
      rightLevels = right_->TreeDepth();
    }
    size_t leftLevels = left_->TreeDepth();
    if (leftLevels > rightLevels)
    {
      levels += leftLevels;
    }
    else
    {
      levels += rightLevels;
    }
  }
  return levels;
}

template<typename Bound, typename Statistic>
inline const Bound& BinarySpaceTree<Bound, Statistic>::bound() const
{
  return bound_;
}

template<typename Bound, typename Statistic>
inline Bound& BinarySpaceTree<Bound, Statistic>::bound()
{
  return bound_;
}

template<typename Bound, typename Statistic>
inline const Statistic& BinarySpaceTree<Bound, Statistic>::stat() const
{
  return stat_;
}

template<typename Bound, typename Statistic>
inline Statistic& BinarySpaceTree<Bound, Statistic>::stat()
{
  return stat_;
}

template<typename Bound, typename Statistic>
inline bool BinarySpaceTree<Bound, Statistic>::is_leaf() const
{
  return !left_;
}

/**
 * Gets the left branch of the tree.
 */
template<typename Bound, typename Statistic>
inline BinarySpaceTree<Bound, Statistic>*
BinarySpaceTree<Bound, Statistic>::left() const
{
  return left_;
}

/**
 * Gets the right branch.
 */
template<typename Bound, typename Statistic>
inline BinarySpaceTree<Bound, Statistic>*
BinarySpaceTree<Bound, Statistic>::right() const
{
  return right_;
}

/**
 * Gets the index of the begin point of this subset.
 */
template<typename Bound, typename Statistic>
inline size_t BinarySpaceTree<Bound, Statistic>::begin() const
{
  return begin_;
}

/**
 * Gets the index one beyond the last index in the series.
 */
template<typename Bound, typename Statistic>
inline size_t BinarySpaceTree<Bound, Statistic>::end() const
{
  return begin_ + count_;
}

/**
 * Gets the number of points in this subset.
 */
template<typename Bound, typename Statistic>
inline size_t BinarySpaceTree<Bound, Statistic>::count() const
{
  return count_;
}

template<typename Bound, typename Statistic>
void BinarySpaceTree<Bound, Statistic>::Print() const {
  printf("node: %d to %d: %d points total\n",
      begin_, begin_ + count_ - 1, count_);
  if (!is_leaf())
  {
    left_->Print();
    if (right_)
    {
      right_->Print();
    }
  }
}

template<typename Bound, typename Statistic>
void BinarySpaceTree<Bound, Statistic>::SplitNode(arma::mat& data)
{
  // This should be a single function for Bound.
  // We need to expand the bounds of this node properly.
  for (size_t i = begin_; i < (begin_ + count_); i++)
    bound_ |= data.unsafe_col(i);

  // Now, check if we need to split at all.
  if (count_ <= (size_t) CLI::GetParam<int>("tree/leaf_size"))
    return; // We can't split this.

  // Figure out which dimension to split on.
  size_t split_dim = data.n_rows; // Indicate invalid by max_dim + 1.
  double max_width = -1;

  // Find the split dimension.
  for (size_t d = 0; d < data.n_rows; d++)
  {
    double width = bound_[d].width();

    if (width > max_width)
    {
      max_width = width;
      split_dim = d;
    }
  }

  if (max_width == 0) // All these points are the same.  We can't split.
    return;

  double split_value = bound_[split_dim].mid();

  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension split_dim less than or equal to split_value are on
  // the left of split_col, and points with value in dimension split_dim greater
  // than split_value are on the right side of split_col.
  size_t split_col = GetSplitIndex(data, split_dim, split_value);
  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left_ = new BinarySpaceTree<Bound, Statistic>(data, begin_,
      split_col - begin_);
  right_ = new BinarySpaceTree<Bound, Statistic>(data, split_col,
      begin_ + count_ - split_col);
}

template<typename Bound, typename Statistic>
void BinarySpaceTree<Bound, Statistic>::SplitNode(
    arma::mat& data,
    std::vector<size_t>& old_from_new)
{
  // This should be a single function for Bound.
  // We need to expand the bounds of this node properly.
  for (size_t i = begin_; i < (begin_ + count_); i++)
    bound_ |= data.unsafe_col(i);

  // First, check if we need to split at all.
  if (count_ <= (size_t) CLI::GetParam<int>("tree/leaf_size"))
    return; // We can't split this.

  // Figure out which dimension to split on.
  size_t split_dim = data.n_rows; // Indicate invalid by max_dim + 1.
  double max_width = -DBL_MAX;

  // Find the split dimension.
  for (size_t d = 0; d < data.n_rows; d++)
  {
    double width = bound_[d].width();

    if (width > max_width)
    {
      max_width = width;
      split_dim = d;
    }
  }

  if (max_width == 0) // All these points are the same.  We can't split.
    return;

  // Split in the middle of that dimension.
  size_t split_col = -1;
  double split_value = -DBL_MAX;

  split_value = bound_[split_dim].mid();
  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension split_dim less than or equal to split_value are on
  // the left of split_col, and points with value in dimension split_dim greater
  // than split_value are on the right side of split_col.
  split_col = GetSplitIndex(data, split_dim, split_value, old_from_new);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left_ = new BinarySpaceTree<Bound, Statistic>(data, begin_,
      split_col - begin_, old_from_new);
  right_ = new BinarySpaceTree<Bound, Statistic>(data, split_col,
      begin_ + count_ - split_col, old_from_new);
}

template<typename Bound, typename Statistic>
size_t BinarySpaceTree<Bound, Statistic>::GetSplitIndex(
    arma::mat& data,
    int split_dim,
    double split_val)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points less than
  // split_val should be on the left side of the matrix, and the points greater
  // than split_val should be on the right side of the matrix.
  size_t left = begin_;
  size_t right = begin_ + count_ - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((data(split_dim, left) < split_val) && (left <= right))
    left++;
  while ((data(split_dim, right) >= split_val) && (left <= right))
    right--;

  while(left <= right)
  {
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

  Log::Assert(left == right + 1);

  return left;
}

template<typename Bound, typename Statistic>
size_t BinarySpaceTree<Bound, Statistic>::GetSplitIndex(
    arma::mat& data,
    int split_dim,
    double split_val,
    std::vector<size_t>& old_from_new)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points less than
  // split_val should be on the left side of the matrix, and the points greater
  // than split_val should be on the right side of the matrix.
  size_t left = begin_;
  size_t right = begin_ + count_ -1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((data(split_dim, left) < split_val) && (left <= right))
    left++;
  while ((data(split_dim, right) >= split_val) && (left <= right))
    right--;

  while(left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // Update the indices for what we changed.
    size_t t = old_from_new[left];
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

  Log::Assert(left == right + 1);

  return left;
}

}; // namespace tree
}; // namespace mlpack

#endif
