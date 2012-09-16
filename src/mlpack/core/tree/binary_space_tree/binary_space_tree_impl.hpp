/**
 * @file binary_space_tree_impl.hpp
 *
 * Implementation of generalized space partitioning tree.
 *
 * This file is part of MLPACK 1.0.3.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_IMPL_HPP
#define __MLPACK_CORE_TREE_BINARY_SPACE_TREE_BINARY_SPACE_TREE_IMPL_HPP

// In case it wasn't included already for some reason.
#include "binary_space_tree.hpp"

#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/log.hpp>

namespace mlpack {
namespace tree {

// Each of these overloads is kept as a separate function to keep the overhead
// from the two std::vectors out, if possible.
template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree(
    MatType& data,
    const size_t leafSize) :
    left(NULL),
    right(NULL),
    begin(0), /* This root node starts at index 0, */
    count(data.n_cols), /* and spans all of the dataset. */
    bound(data.n_rows),
    stat(),
    leafSize(leafSize)
{
  // Do the actual splitting of this node.
  SplitNode(data);

  // Create the statistic depending on if we are a leaf or not.
  if (IsLeaf())
    stat = StatisticType(data, begin, count);
  else
    stat = StatisticType(data, begin, count, left->Stat(), right->Stat());
}

template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree(
    MatType& data,
    std::vector<size_t>& oldFromNew,
    const size_t leafSize) :
    left(NULL),
    right(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    stat(),
    leafSize(leafSize)
{
  // Initialize oldFromNew correctly.
  oldFromNew.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    oldFromNew[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitNode(data, oldFromNew);

  // Create the statistic depending on if we are a leaf or not.
  if (IsLeaf())
    stat = StatisticType(data, begin, count);
  else
    stat = StatisticType(data, begin, count, left->Stat(), right->Stat());
}

template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree(
    MatType& data,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    const size_t leafSize) :
    left(NULL),
    right(NULL),
    begin(0),
    count(data.n_cols),
    bound(data.n_rows),
    stat(),
    leafSize(leafSize)
{
  // Initialize the oldFromNew vector correctly.
  oldFromNew.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    oldFromNew[i] = i; // Fill with unharmed indices.

  // Now do the actual splitting.
  SplitNode(data, oldFromNew);

  // Create the statistic depending on if we are a leaf or not.
  if (IsLeaf())
    stat = StatisticType(data, begin, count);
  else
    stat = StatisticType(data, begin, count, left->Stat(), right->Stat());

  // Map the newFromOld indices correctly.
  newFromOld.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    newFromOld[oldFromNew[i]] = i;
}

template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree(
    MatType& data,
    const size_t begin,
    const size_t count,
    const size_t leafSize) :
    left(NULL),
    right(NULL),
    begin(begin),
    count(count),
    bound(data.n_rows),
    stat(),
    leafSize(leafSize)
{
  // Perform the actual splitting.
  SplitNode(data);

  // Create the statistic depending on if we are a leaf or not.
  if (IsLeaf())
    stat = StatisticType(data, begin, count);
  else
    stat = StatisticType(data, begin, count, left->Stat(), right->Stat());
}

template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree(
    MatType& data,
    const size_t begin,
    const size_t count,
    std::vector<size_t>& oldFromNew,
    const size_t leafSize) :
    left(NULL),
    right(NULL),
    begin(begin),
    count(count),
    bound(data.n_rows),
    stat(),
    leafSize(leafSize)
{
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  assert(oldFromNew.size() == data.n_cols);

  // Perform the actual splitting.
  SplitNode(data, oldFromNew);

  // Create the statistic depending on if we are a leaf or not.
  if (IsLeaf())
    stat = StatisticType(data, begin, count);
  else
    stat = StatisticType(data, begin, count, left->Stat(), right->Stat());
}

template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree(
    MatType& data,
    const size_t begin,
    const size_t count,
    std::vector<size_t>& oldFromNew,
    std::vector<size_t>& newFromOld,
    const size_t leafSize) :
    left(NULL),
    right(NULL),
    begin(begin),
    count(count),
    bound(data.n_rows),
    stat(),
    leafSize(leafSize)
{
  // Hopefully the vector is initialized correctly!  We can't check that
  // entirely but we can do a minor sanity check.
  Log::Assert(oldFromNew.size() == data.n_cols);

  // Perform the actual splitting.
  SplitNode(data, oldFromNew);

  // Create the statistic depending on if we are a leaf or not.
  if (IsLeaf())
    stat = StatisticType(data, begin, count);
  else
    stat = StatisticType(data, begin, count, left->Stat(), right->Stat());

  // Map the newFromOld indices correctly.
  newFromOld.resize(data.n_cols);
  for (size_t i = 0; i < data.n_cols; i++)
    newFromOld[oldFromNew[i]] = i;
}

template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::BinarySpaceTree() :
    left(NULL),
    right(NULL),
    begin(0),
    count(0),
    bound(),
    stat(),
    leafSize(20) // Default leaf size is 20.
{
  // Nothing to do.
}

/**
 * Deletes this node, deallocating the memory for the children and calling their
 * destructors in turn.  This will invalidate any pointers or references to any
 * nodes which are children of this one.
 */
template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>::~BinarySpaceTree()
{
  if (left)
    delete left;
  if (right)
    delete right;
}

/**
 * Find a node in this tree by its begin and count.
 *
 * Every node is uniquely identified by these two numbers.
 * This is useful for communicating position over the network,
 * when pointers would be invalid.
 *
 * @param queryBegin The Begin() of the node to find.
 * @param queryCount The Count() of the node to find.
 * @return The found node, or NULL if nothing is found.
 */
template<typename BoundType, typename StatisticType, typename MatType>
const BinarySpaceTree<BoundType, StatisticType, MatType>*
BinarySpaceTree<BoundType, StatisticType, MatType>::FindByBeginCount(
    size_t queryBegin,
    size_t queryCount) const
{
  Log::Assert(queryBegin >= begin);
  Log::Assert(queryCount <= count);

  if (begin == queryBegin && count == queryCount)
    return this;
  else if (IsLeaf())
    return NULL;
  else if (queryBegin < right->Begin())
    return left->FindByBeginCount(queryBegin, queryCount);
  else
    return right->FindByBeginCount(queryBegin, queryCount);
}

/**
 * Find a node in this tree by its begin and count (const).
 *
 * Every node is uniquely identified by these two numbers.
 * This is useful for communicating position over the network,
 * when pointers would be invalid.
 *
 * @param queryBegin the Begin() of the node to find
 * @param queryCount the Count() of the node to find
 * @return the found node, or NULL
 */
template<typename BoundType, typename StatisticType, typename MatType>
BinarySpaceTree<BoundType, StatisticType, MatType>*
BinarySpaceTree<BoundType, StatisticType, MatType>::FindByBeginCount(
    const size_t queryBegin,
    const size_t queryCount)
{
  mlpack::Log::Assert(begin >= queryBegin);
  mlpack::Log::Assert(count <= queryCount);

  if (begin == queryBegin && count == queryCount)
    return this;
  else if (IsLeaf())
    return NULL;
  else if (queryBegin < left->End())
    return left->FindByBeginCount(queryBegin, queryCount);
  else if (right)
    return right->FindByBeginCount(queryBegin, queryCount);
  else
    return NULL;
}

template<typename BoundType, typename StatisticType, typename MatType>
size_t BinarySpaceTree<BoundType, StatisticType, MatType>::ExtendTree(
    size_t level)
{
  --level;
  // Return the number of nodes duplicated.
  size_t nodesDuplicated = 0;
  if (level > 0)
  {
    if (!left)
    {
      left = CopyMe();
      ++nodesDuplicated;
    }
    nodesDuplicated += left->ExtendTree(level);
    if (right)
    {
      nodesDuplicated += right->ExtendTree(level);
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

template<typename BoundType, typename StatisticType, typename MatType>
size_t BinarySpaceTree<BoundType, StatisticType, MatType>::TreeSize() const
{
  // Recursively count the nodes on each side of the tree.  The plus one is
  // because we have to count this node, too.
  return 1 + (left ? left->TreeSize() : 0) + (right ? right->TreeSize() : 0);
}

template<typename BoundType, typename StatisticType, typename MatType>
size_t BinarySpaceTree<BoundType, StatisticType, MatType>::TreeDepth() const
{
  // Recursively count the depth on each side of the tree.  The plus one is
  // because we have to count this node, too.
  return 1 + std::max((left ? left->TreeDepth() : 0),
                      (right ? right->TreeDepth() : 0));
}

template<typename BoundType, typename StatisticType, typename MatType>
inline const
    BoundType& BinarySpaceTree<BoundType, StatisticType, MatType>::Bound() const
{
  return bound;
}

template<typename BoundType, typename StatisticType, typename MatType>
inline BoundType& BinarySpaceTree<BoundType, StatisticType, MatType>::Bound()
{
  return bound;
}

template<typename BoundType, typename StatisticType, typename MatType>
inline const StatisticType&
    BinarySpaceTree<BoundType, StatisticType, MatType>::Stat() const
{
  return stat;
}

template<typename BoundType, typename StatisticType, typename MatType>
inline StatisticType& BinarySpaceTree<BoundType, StatisticType, MatType>::Stat()
{
  return stat;
}

template<typename BoundType, typename StatisticType, typename MatType>
inline size_t BinarySpaceTree<BoundType, StatisticType, MatType>::GetSplitDimension() const
{
  return splitDimension;
}

template<typename BoundType, typename StatisticType, typename MatType>
inline bool BinarySpaceTree<BoundType, StatisticType, MatType>::IsLeaf() const
{
  return !left;
}

/**
 * Gets the left branch of the tree.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline BinarySpaceTree<BoundType, StatisticType, MatType>*
    BinarySpaceTree<BoundType, StatisticType, MatType>::Left() const
{
  return left;
}

/**
 * Gets the right branch.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline BinarySpaceTree<BoundType, StatisticType, MatType>*
    BinarySpaceTree<BoundType, StatisticType, MatType>::Right() const
{
  return right;
}

/**
 * Returns the number of children in this node.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline size_t
    BinarySpaceTree<BoundType, StatisticType, MatType>::NumChildren() const
{
  if (left && right)
    return 2;
  if (left)
    return 1;

  return 0;
}

/**
 * Return the specified child.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline BinarySpaceTree<BoundType, StatisticType, MatType>&
    BinarySpaceTree<BoundType, StatisticType, MatType>::Child(
    const size_t child) const
{
  if (child == 0)
    return *left;
  else
    return *right;
}

/**
 * Return the number of points contained in this node.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline size_t
BinarySpaceTree<BoundType, StatisticType, MatType>::NumPoints() const
{
  if (left)
    return 0;

  return count;
}

/**
 * Return the index of a particular point contained in this node.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline size_t
BinarySpaceTree<BoundType, StatisticType, MatType>::Point(const size_t index)
    const
{
  return (begin + index);
}

/**
 * Gets the index of the begin point of this subset.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline size_t BinarySpaceTree<BoundType, StatisticType, MatType>::Begin() const
{
  return begin;
}

/**
 * Gets the index one beyond the last index in the series.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline size_t BinarySpaceTree<BoundType, StatisticType, MatType>::End() const
{
  return begin + count;
}

/**
 * Gets the number of points in this subset.
 */
template<typename BoundType, typename StatisticType, typename MatType>
inline size_t BinarySpaceTree<BoundType, StatisticType, MatType>::Count() const
{
  return count;
}

template<typename BoundType, typename StatisticType, typename MatType>
void
    BinarySpaceTree<BoundType, StatisticType, MatType>::SplitNode(MatType& data)
{
  // We need to expand the bounds of this node properly.
  bound |= data.cols(begin, begin + count - 1);

  // Now, check if we need to split at all.
  if (count <= leafSize)
    return; // We can't split this.

  // Figure out which dimension to split on.
  size_t splitDim = data.n_rows; // Indicate invalid by maxDim + 1.
  double maxWidth = -1;

  // Find the split dimension.
  for (size_t d = 0; d < data.n_rows; d++)
  {
    double width = bound[d].Width();

    if (width > maxWidth)
    {
      maxWidth = width;
      splitDim = d;
    }
  }
  splitDimension = splitDim;

  // Split in the middle of that dimension.
  double splitVal = bound[splitDim].Mid();

  if (maxWidth == 0) // All these points are the same.  We can't split.
    return;

  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension split_dim less than or equal to splitVal are on
  // the left of splitCol, and points with value in dimension splitDim greater
  // than splitVal are on the right side of splitCol.
  size_t splitCol = GetSplitIndex(data, splitDim, splitVal);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left = new BinarySpaceTree<BoundType, StatisticType, MatType>(data, begin,
      splitCol - begin, leafSize);
  right = new BinarySpaceTree<BoundType, StatisticType, MatType>(data, splitCol,
      begin + count - splitCol, leafSize);
}

template<typename BoundType, typename StatisticType, typename MatType>
void BinarySpaceTree<BoundType, StatisticType, MatType>::SplitNode(
    MatType& data,
    std::vector<size_t>& oldFromNew)
{
  // This should be a single function for Bound.
  // We need to expand the bounds of this node properly.
  bound |= data.cols(begin, begin + count - 1);

  // First, check if we need to split at all.
  if (count <= leafSize)
    return; // We can't split this.

  // Figure out which dimension to split on.
  size_t splitDim = data.n_rows; // Indicate invalid by max_dim + 1.
  double maxWidth = -1;

  // Find the split dimension.
  for (size_t d = 0; d < data.n_rows; d++)
  {
    double width = bound[d].Width();

    if (width > maxWidth)
    {
      maxWidth = width;
      splitDim = d;
    }
  }
  splitDimension = splitDim;

  // Split in the middle of that dimension.
  double splitVal = bound[splitDim].Mid();

  if (maxWidth == 0) // All these points are the same.  We can't split.
    return;

  // Perform the actual splitting.  This will order the dataset such that points
  // with value in dimension split_dim less than or equal to splitVal are on
  // the left of splitCol, and points with value in dimension splitDim greater
  // than splitVal are on the right side of splitCol.
  size_t splitCol = GetSplitIndex(data, splitDim, splitVal, oldFromNew);

  // Now that we know the split column, we will recursively split the children
  // by calling their constructors (which perform this splitting process).
  left = new BinarySpaceTree<BoundType, StatisticType, MatType>(data, begin,
      splitCol - begin, oldFromNew, leafSize);
  right = new BinarySpaceTree<BoundType, StatisticType, MatType>(data, splitCol,
      begin + count - splitCol, oldFromNew, leafSize);
}

template<typename BoundType, typename StatisticType, typename MatType>
size_t BinarySpaceTree<BoundType, StatisticType, MatType>::GetSplitIndex(
    MatType& data,
    int splitDim,
    double splitVal)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points less than
  // split_val should be on the left side of the matrix, and the points greater
  // than split_val should be on the right side of the matrix.
  size_t left = begin;
  size_t right = begin + count - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((data(splitDim, left) < splitVal) && (left <= right))
    left++;
  while ((data(splitDim, right) >= splitVal) && (left <= right))
    right--;

  while (left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while ((data(splitDim, left) < splitVal) && (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((data(splitDim, right) >= splitVal) && (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

template<typename BoundType, typename StatisticType, typename MatType>
size_t BinarySpaceTree<BoundType, StatisticType, MatType>::GetSplitIndex(
    MatType& data,
    int splitDim,
    double splitVal,
    std::vector<size_t>& oldFromNew)
{
  // This method modifies the input dataset.  We loop both from the left and
  // right sides of the points contained in this node.  The points less than
  // split_val should be on the left side of the matrix, and the points greater
  // than split_val should be on the right side of the matrix.
  size_t left = begin;
  size_t right = begin + count - 1;

  // First half-iteration of the loop is out here because the termination
  // condition is in the middle.
  while ((data(splitDim, left) < splitVal) && (left <= right))
    left++;
  while ((data(splitDim, right) >= splitVal) && (left <= right))
    right--;

  while (left <= right)
  {
    // Swap columns.
    data.swap_cols(left, right);

    // Update the indices for what we changed.
    size_t t = oldFromNew[left];
    oldFromNew[left] = oldFromNew[right];
    oldFromNew[right] = t;

    // See how many points on the left are correct.  When they are correct,
    // increase the left counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it later.
    while ((data(splitDim, left) < splitVal) && (left <= right))
      left++;

    // Now see how many points on the right are correct.  When they are correct,
    // decrease the right counter accordingly.  When we encounter one that isn't
    // correct, stop.  We will switch it with the wrong point we found in the
    // previous loop.
    while ((data(splitDim, right) >= splitVal) && (left <= right))
      right--;
  }

  Log::Assert(left == right + 1);

  return left;
}

}; // namespace tree
}; // namespace mlpack

#endif
